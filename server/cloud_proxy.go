package server

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/klauspost/compress/zstd"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/auth"
	"github.com/ollama/ollama/envconfig"
	internalcloud "github.com/ollama/ollama/internal/cloud"
	"github.com/ollama/ollama/types/model"
	"github.com/ollama/ollama/version"
)

const (
	defaultCloudProxyBaseURL      = "https://ollama.com:443"
	defaultCloudProxySigningHost  = "ollama.com"
	cloudProxyBaseURLEnv          = "OLLAMA_CLOUD_BASE_URL"
	legacyCloudAnthropicKey       = "legacy_cloud_anthropic_web_search"
	cloudAnthropicImageKey        = "cloud_anthropic_image"
	cloudProxyClientVersionHeader = "X-Ollama-Client-Version"

	// maxDecompressedBodySize limits the size of a decompressed request body
	maxDecompressedBodySize = 20 << 20
)

var (
	cloudProxyBaseURL     = defaultCloudProxyBaseURL
	cloudProxySigningHost = defaultCloudProxySigningHost
	cloudProxySignRequest = signCloudProxyRequest
	cloudProxySigninURL   = signinURL
)

var hopByHopHeaders = map[string]struct{}{
	"connection":          {},
	"content-length":      {},
	"proxy-connection":    {},
	"keep-alive":          {},
	"proxy-authenticate":  {},
	"proxy-authorization": {},
	"te":                  {},
	"trailer":             {},
	"transfer-encoding":   {},
	"upgrade":             {},
}

func init() {
	baseURL, signingHost, overridden, err := resolveCloudProxyBaseURL(envconfig.Var(cloudProxyBaseURLEnv), mode)
	if err != nil {
		slog.Warn("ignoring cloud base URL override", "env", cloudProxyBaseURLEnv, "error", err)
		return
	}

	cloudProxyBaseURL = baseURL
	cloudProxySigningHost = signingHost

	if overridden {
		slog.Info("cloud base URL override enabled", "env", cloudProxyBaseURLEnv, "url", cloudProxyBaseURL, "mode", mode)
	}
}

func cloudPassthroughMiddleware(disabledOperation string) gin.HandlerFunc {
	return func(c *gin.Context) {
		if c.Request.Method != http.MethodPost {
			c.Next()
			return
		}

		// Decompress zstd-encoded request bodies so we can inspect the model
		if c.GetHeader("Content-Encoding") == "zstd" {
			reader, err := zstd.NewReader(c.Request.Body, zstd.WithDecoderMaxMemory(8<<20))
			if err != nil {
				c.JSON(http.StatusBadRequest, gin.H{"error": "failed to decompress request body"})
				c.Abort()
				return
			}
			defer reader.Close()
			c.Request.Body = http.MaxBytesReader(c.Writer, io.NopCloser(reader), maxDecompressedBodySize)
			c.Request.Header.Del("Content-Encoding")
		}

		// TODO(drifkin): Avoid full-body buffering here for model detection.
		// A future optimization can parse just enough JSON to read "model" (and
		// optionally short-circuit cloud-disabled explicit-cloud requests) while
		// preserving raw passthrough semantics.
		body, err := readRequestBody(c.Request)
		if err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
			c.Abort()
			return
		}

		model, ok := extractModelField(body)
		if !ok {
			c.Next()
			return
		}

		modelRef, err := parseAndValidateModelRef(model)
		if err != nil || modelRef.Source != modelSourceCloud {
			c.Next()
			return
		}

		normalizedBody, err := replaceJSONModelField(body, modelRef.Base)
		if err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
			c.Abort()
			return
		}

		// TEMP(drifkin): keep Anthropic web search requests on the local middleware
		// path so WebSearchAnthropicWriter can orchestrate follow-up calls.
		if c.Request.URL.Path == "/v1/messages" {
			if hasAnthropicWebSearchTool(body) {
				c.Set(legacyCloudAnthropicKey, true)
				c.Next()
				return
			}

			// Cloud /v1/messages is raw-proxied to the remote Anthropic
			// endpoint, which does not accept Anthropic image content blocks
			// for cloud models (it returns "this model does not support image
			// input", even for image-capable models). Divert image-bearing
			// requests to the local converter path so they are translated to
			// Ollama /api/chat format (which the cloud accepts) and the
			// response is translated back to Anthropic SSE. This also enables
			// the OLLAMA_CLOUD_VISION_FALLBACK retry in ChatHandler.
			if hasAnthropicImageContent(body) {
				c.Set(cloudAnthropicImageKey, true)
				c.Next()
				return
			}
		}

		proxyCloudRequest(c, normalizedBody, disabledOperation)
		c.Abort()
	}
}

func cloudModelPathPassthroughMiddleware(disabledOperation string) gin.HandlerFunc {
	return func(c *gin.Context) {
		modelName := strings.TrimSpace(c.Param("model"))
		if modelName == "" {
			c.Next()
			return
		}

		modelRef, err := parseAndValidateModelRef(modelName)
		if err != nil || modelRef.Source != modelSourceCloud {
			c.Next()
			return
		}

		proxyPath := "/v1/models/" + modelRef.Base
		proxyCloudRequestWithPath(c, nil, proxyPath, disabledOperation)
		c.Abort()
	}
}

func proxyCloudJSONRequest(c *gin.Context, payload any, disabledOperation string) {
	// TEMP(drifkin): we currently split out this `WithPath` method because we are
	// mapping `/v1/messages` + web_search to `/api/chat` temporarily. Once we
	// stop doing this, we can inline this method.
	proxyCloudJSONRequestWithPath(c, payload, c.Request.URL.Path, disabledOperation)
}

func proxyCloudJSONRequestWithPath(c *gin.Context, payload any, path string, disabledOperation string) {
	body, err := json.Marshal(payload)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	proxyCloudRequestWithPath(c, body, path, disabledOperation)
}

func proxyCloudRequest(c *gin.Context, body []byte, disabledOperation string) {
	proxyCloudRequestWithPath(c, body, c.Request.URL.Path, disabledOperation)
}

func proxyCloudRequestWithPath(c *gin.Context, body []byte, path string, disabledOperation string) {
	resp, err := executeCloudProxyRequest(c, body, path, disabledOperation)
	if err != nil {
		respondCloudProxyError(c, err, disabledOperation)
		return
	}
	defer resp.Body.Close()
	streamCloudResponse(c, resp, path)
}

// executeCloudProxyRequest builds, signs, and sends a proxied request to the
// cloud at the given path. The caller is responsible for closing resp.Body.
// It does not write to c on failure; callers should pass the returned error to
// respondCloudProxyError (or handle it themselves for sub-requests that must not
// touch the client response).
func executeCloudProxyRequest(c *gin.Context, body []byte, path, disabledOperation string) (*http.Response, error) {
	if disabled, _ := internalcloud.Status(); disabled {
		return nil, errCloudDisabled
	}

	baseURL, err := url.Parse(cloudProxyBaseURL)
	if err != nil {
		return nil, fmt.Errorf("invalid cloud base url: %w", err)
	}

	targetURL := baseURL.ResolveReference(&url.URL{
		Path:     path,
		RawQuery: c.Request.URL.RawQuery,
	})

	outReq, err := http.NewRequestWithContext(c.Request.Context(), c.Request.Method, targetURL.String(), bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("build cloud request: %w", err)
	}

	copyProxyRequestHeaders(outReq.Header, c.Request.Header)
	if clientVersion := strings.TrimSpace(version.Version); clientVersion != "" {
		outReq.Header.Set(cloudProxyClientVersionHeader, clientVersion)
	}
	if outReq.Header.Get("Content-Type") == "" && len(body) > 0 {
		outReq.Header.Set("Content-Type", "application/json")
	}

	if err := cloudProxySignRequest(outReq.Context(), outReq); err != nil {
		slog.Warn("cloud proxy signing failed", "error", err)
		return nil, errCloudSigning
	}

	// TODO(drifkin): Add phase-specific proxy timeouts.
	// Connect/TLS/TTFB should have bounded timeouts, but once streaming starts
	// we should not enforce a short total timeout for long-lived responses.
	resp, err := http.DefaultClient.Do(outReq)
	if err != nil {
		return nil, fmt.Errorf("%w: %v", errCloudTransport, err)
	}
	return resp, nil
}

// respondCloudProxyError writes the appropriate client response for an
// executeCloudProxyRequest setup failure, mirroring the original per-case
// status codes (403 disabled, 401 signing, 502 transport, 500 otherwise).
func respondCloudProxyError(c *gin.Context, err error, disabledOperation string) {
	switch {
	case errors.Is(err, errCloudDisabled):
		c.JSON(http.StatusForbidden, gin.H{"error": internalcloud.DisabledError(disabledOperation)})
	case errors.Is(err, errCloudSigning):
		writeCloudUnauthorized(c)
	case errors.Is(err, errCloudTransport):
		c.JSON(http.StatusBadGateway, gin.H{"error": err.Error()})
	default:
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
	}
}

var (
	errCloudDisabled  = errors.New("cloud features disabled")
	errCloudSigning   = errors.New("cloud proxy signing failed")
	errCloudTransport = errors.New("cloud proxy request failed")
)

// streamCloudResponse copies a cloud response through to the client. It applies
// jsonl framing when proxying a 200 /api/chat response on a diverted Anthropic
// path (legacy web_search or image), since the downstream writer
// (WebSearchAnthropicWriter / AnthropicWriter) unmarshals one JSON value per
// Write and the proxy copy loop may otherwise coalesce multiple jsonl records
// into a single Write.
func streamCloudResponse(c *gin.Context, resp *http.Response, path string) {
	copyProxyResponseHeaders(c.Writer.Header(), resp.Header)
	c.Status(resp.StatusCode)

	var bodyWriter http.ResponseWriter = c.Writer
	var framedWriter *jsonlFramingResponseWriter
	if path == "/api/chat" && resp.StatusCode == http.StatusOK && (c.GetBool(legacyCloudAnthropicKey) || c.GetBool(cloudAnthropicImageKey)) {
		framedWriter = &jsonlFramingResponseWriter{ResponseWriter: c.Writer}
		bodyWriter = framedWriter
	}

	err := copyProxyResponseBody(bodyWriter, resp.Body)
	if err == nil && framedWriter != nil {
		err = framedWriter.FlushPending()
	}
	if err != nil {
		ctxErr := c.Request.Context().Err()
		if errors.Is(err, context.Canceled) && errors.Is(ctxErr, context.Canceled) {
			slog.Debug(
				"cloud proxy response stream closed by client",
				"path", c.Request.URL.Path,
				"status", resp.StatusCode,
			)
			return
		}

		slog.Warn(
			"cloud proxy response copy failed",
			"path", c.Request.URL.Path,
			"upstream_path", path,
			"status", resp.StatusCode,
			"request_context_canceled", ctxErr != nil,
			"request_context_err", ctxErr,
			"error", err,
		)
		return
	}
}

// proxyCloudChatWithVisionFallback proxies a converted Ollama /api/chat request
// to the cloud for a diverted Anthropic /v1/messages image request.
//
// Behavior:
//   - With no OLLAMA_CLOUD_VISION_FALLBACK configured, the request (images and
//     all) goes to the primary model. An image-capable primary handles it; a
//     non-vision primary rejects it and the error surfaces.
//   - With a fallback configured, the primary is tried first with the real
//     image bytes. If it accepts (image-capable primary), its response is
//     streamed. If it rejects with "does not support image input", the fallback
//     model is used only to caption each image (cached by image hash); the
//     captions replace the image bytes in the conversation and the request is
//     re-sent to the primary as text. This keeps the primary as the brain for
//     the whole task while letting a non-vision primary "see" images via the
//     caption. The primary is remembered as non-vision so later turns skip the
//     doomed first attempt, and captions are cached so later turns are a single
//     primary call.
//   - If captioning fails, the original image request is sent to the fallback
//     model as a graceful degradation so the turn still gets an answer.
func proxyCloudChatWithVisionFallback(c *gin.Context, req api.ChatRequest, disabledOperation string) {
	fallback := strings.TrimSpace(envconfig.CloudVisionFallback())
	// A per-launch fallback (set via `ollama launch --fallback`, conveyed as a
	// request header by the launch shim) overrides the server-wide env var so
	// concurrent launches can use different fallbacks against one server.
	if h := strings.TrimSpace(c.GetHeader("X-Ollama-Cloud-Vision-Fallback")); h != "" {
		fallback = h
	}
	if fallback == "" {
		proxyCloudJSONRequestWithPath(c, req, "/api/chat", disabledOperation)
		return
	}

	fallbackRef, err := parseAndValidateModelRef(fallback)
	if err != nil {
		slog.Warn("invalid cloud vision fallback model, ignoring", "value", fallback, "error", err)
		proxyCloudJSONRequestWithPath(c, req, "/api/chat", disabledOperation)
		return
	}

	// Try the primary with the real image bytes first, unless we already know
	// it cannot handle images. This lets image-capable primaries see the
	// pixels directly (no lossy caption). We know the primary can't handle
	// images either from a prior rejection (nonVisionModels) or proactively
	// from /api/show reporting no vision capability — the proactive check
	// avoids a wasted doomed attempt on the first image turn. When /api/show
	// doesn't report capabilities we fall back to trying and catching the 400.
	primaryNonVision := nonVisionModels.isNonVision(req.Model)
	if !primaryNonVision {
		if caps, known := cloudModelCapabilities(c, req.Model, disabledOperation); known && !hasCapability(caps, model.CapabilityVision) {
			primaryNonVision = true
			nonVisionModels.mark(req.Model)
			slog.Info("cloud model has no vision capability, captioning via fallback then retrying primary",
				"model", req.Model, "fallback", fallbackRef.Base)
		}
	}
	if !primaryNonVision {
		body, err := json.Marshal(req)
		if err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
			return
		}
		resp, err := executeCloudProxyRequest(c, body, "/api/chat", disabledOperation)
		if err != nil {
			respondCloudProxyError(c, err, disabledOperation)
			return
		}

		if resp.StatusCode == http.StatusOK {
			defer resp.Body.Close()
			streamCloudResponse(c, resp, "/api/chat")
			return
		}

		if resp.StatusCode != http.StatusBadRequest {
			defer resp.Body.Close()
			streamCloudResponse(c, resp, "/api/chat")
			return
		}

		// 400: buffer it so we can inspect and retry without committing bytes.
		errBody, readErr := io.ReadAll(io.LimitReader(resp.Body, maxDecompressedBodySize))
		resp.Body.Close()
		if readErr != nil {
			c.JSON(http.StatusBadGateway, gin.H{"error": readErr.Error()})
			return
		}

		if !isImageNotSupportedError(errBody) {
			// Unrelated 400: surface it (AnthropicWriter translates non-200).
			copyProxyResponseHeaders(c.Writer.Header(), resp.Header)
			c.Status(http.StatusBadRequest)
			if _, err := c.Writer.Write(errBody); err != nil {
				slog.Warn("cloud proxy error body write failed", "error", err)
			}
			return
		}

		// Primary rejected the image. Remember so later turns skip this attempt.
		nonVisionModels.mark(req.Model)
		slog.Info("cloud model does not support image input, captioning via fallback then retrying primary",
			"model", req.Model, "fallback", fallbackRef.Base)
		// Fall through to caption-then-primary below.
	}

	// Caption every image (cached), replace image bytes with caption text, and
	// re-send to the primary as text. On caption failure, degrade to letting
	// the fallback model answer this turn directly.
	if !captionRequestImages(c, &req, fallbackRef.Base, disabledOperation) {
		slog.Warn("image captioning failed, degrading to vision fallback model for this turn", "fallback", fallbackRef.Base)
		req.Model = fallbackRef.Base
		proxyCloudJSONRequestWithPath(c, req, "/api/chat", disabledOperation)
		return
	}

	proxyCloudJSONRequestWithPath(c, req, "/api/chat", disabledOperation)
}

// captionRequestImages replaces every image in req with a text caption
// produced by the fallback vision model, captioning each image-bearing
// message in the full conversation context it appeared in. It captions all
// images first and only mutates req once every caption succeeds, so a failure
// leaves req unchanged. Returns false (with req unmodified) if any image
// could not be captioned.
func captionRequestImages(c *gin.Context, req *api.ChatRequest, fallbackBase, disabledOperation string) bool {
	type msgCaption struct {
		idx  int
		text string
	}
	var captions []msgCaption
	for i := range req.Messages {
		if len(req.Messages[i].Images) == 0 {
			continue
		}
		caption, err := captionMessageInContext(c, req, i, fallbackBase, disabledOperation)
		if err != nil {
			slog.Warn("failed to caption image", "error", err)
			return false
		}
		captions = append(captions, msgCaption{i, caption})
	}

	for _, cap := range captions {
		var b strings.Builder
		b.WriteString("[The user attached an image. A vision model (")
		b.WriteString(fallbackBase)
		b.WriteString(") was given the full conversation up to this point — exactly as if it were the primary model — and responded:\n")
		b.WriteString(cap.text)
		b.WriteString("\n]\n")
		b.WriteString(req.Messages[cap.idx].Content)
		req.Messages[cap.idx].Content = b.String()
		req.Messages[cap.idx].Images = nil
	}
	return true
}

// captionMessageInContext asks the fallback vision model to handle the
// image(s) in req.Messages[msgIdx] by routing it the same request the primary
// would have received up to that point: the system prompt, all prior turns,
// and this message (text + image), with the model swapped to the fallback. In
// other words, the fallback sees the image with exactly the context a native
// vision model would have had at the moment the image was introduced, so its
// response reflects the full task context — not a context-free "describe
// everything" prompt. That response becomes the caption the primary reasons
// over.
//
// Only messages up to and including msgIdx are sent (not the whole live
// request): the Anthropic Messages API is stateless and the client re-sends
// history verbatim each turn, so everything up to the image is fixed across
// turns while the turns after it grow. Trimming here keeps the caption cache
// stable, so later turns hit the cache and cost a single primary call. The
// captioner is not given tools, since its job is to look at the image and
// respond, not to take actions. It issues its own cloud /api/chat request and
// must not write to the client response.
func captionMessageInContext(c *gin.Context, req *api.ChatRequest, msgIdx int, fallbackBase, disabledOperation string) (string, error) {
	contextMsgs := req.Messages[:msgIdx+1]
	key := captionContextKey(contextMsgs)
	if caption, ok := imageCaptionCache.get(key); ok {
		return caption, nil
	}

	// Reuse the primary's options but cap the response length so a caption
	// can't run away with the primary's (potentially large) max_tokens.
	options := make(map[string]any, len(req.Options)+1)
	for k, v := range req.Options {
		options[k] = v
	}
	options["num_predict"] = maxCaptionTokens

	truncateTrue := true
	streamFalse := false

	// The fallback may have a smaller context window than the primary, so a
	// conversation that fit the primary may not fit the fallback. Send the full
	// conversation first (Truncate hints the server to trim if it can); if the
	// fallback still reports its context is full, progressively drop the oldest
	// non-system turns — preserving the system prompt and the image-bearing
	// message — and retry. This mirrors how a real primary would have had its
	// history compacted by the client as it neared the window, so the captioner
	// still sees the image with as much of the intended context as fits. The
	// cache key is taken from the full context, so later turns (which re-send
	// that same fixed history) still hit the cache after the same trim.
	msgs := contextMsgs
	// Proactively trim the conversation to the fallback's context window before
	// sending, so we don't rely on the cloud's default/truncation behavior. The
	// image-bearing message is the last in contextMsgs, so it is never dropped;
	// the system prompt (if present) is also kept. Cloud models don't expose a
	// tokenizer, so this uses a conservative estimate; the reactive
	// trim-on-overflow retry below handles any underestimate.
	if ctxLen := cloudModelContextLength(c, fallbackBase, disabledOperation); ctxLen > 0 {
		for estimateContextTokens(msgs) > ctxLen {
			next, ok := dropOldestNonSystem(msgs)
			if !ok {
				break
			}
			msgs = next
		}
	}
	var caption string
	for {
		captionReq := api.ChatRequest{
			Model:    fallbackBase,
			Stream:   &streamFalse,
			Messages: msgs,
			Options:  options,
			Truncate: &truncateTrue,
		}
		body, err := json.Marshal(captionReq)
		if err != nil {
			return "", err
		}
		resp, err := executeCloudProxyRequest(c, body, "/api/chat", disabledOperation)
		if err != nil {
			return "", err
		}
		respBody, rerr := io.ReadAll(io.LimitReader(resp.Body, maxDecompressedBodySize))
		resp.Body.Close()
		if rerr != nil {
			return "", rerr
		}
		if resp.StatusCode == http.StatusOK {
			var chatResp api.ChatResponse
			if err := json.Unmarshal(respBody, &chatResp); err != nil {
				return "", fmt.Errorf("decode caption response: %w", err)
			}
			caption = strings.TrimSpace(chatResp.Message.Content)
			if caption == "" {
				return "", errors.New("caption model returned empty response")
			}
			break
		}
		if !isContextLengthError(respBody) {
			return "", fmt.Errorf("caption model %q returned status %d: %s", fallbackBase, resp.StatusCode, strings.TrimSpace(string(respBody)))
		}
		next, ok := dropOldestNonSystem(msgs)
		if !ok {
			return "", fmt.Errorf("caption model %q context too small even for system + image message", fallbackBase)
		}
		msgs = next
	}

	imageCaptionCache.put(key, caption)
	return caption, nil
}

// isContextLengthError reports whether body is a context-window / prompt-too-long
// error from the caption model, signalling that the caption request should be
// trimmed and retried rather than treated as a hard failure.
func isContextLengthError(body []byte) bool {
	s := strings.ToLower(string(body))
	switch {
	case strings.Contains(s, "context length"),
		strings.Contains(s, "context window"),
		strings.Contains(s, "maximum context"),
		strings.Contains(s, "context limit"),
		strings.Contains(s, "exceeds the context"),
		strings.Contains(s, "prompt is too long"),
		strings.Contains(s, "input is too long"):
		return true
	}
	return false
}

// dropOldestNonSystem returns msgs with the oldest non-system message removed,
// preserving the leading system prompt (if any) and the trailing image-bearing
// message. It returns ok=false when no message can be dropped without removing
// the system prompt or the final (image) message.
func dropOldestNonSystem(msgs []api.Message) ([]api.Message, bool) {
	if len(msgs) <= 1 {
		return nil, false
	}
	start := 0
	if msgs[0].Role == "system" {
		start = 1
	}
	// Never drop the final (image-bearing) message.
	if start >= len(msgs)-1 {
		return nil, false
	}
	out := make([]api.Message, 0, len(msgs)-1)
	out = append(out, msgs[:start]...)
	out = append(out, msgs[start+1:]...)
	return out, true
}

// maxCaptionTokens caps the fallback's response length when captioning, so a
// caption can't grow to the primary's (potentially very large) max_tokens.
const maxCaptionTokens = 2048

// captionContextKey returns a stable cache key for captioning the image(s) in
// the last message of msgs: a hash over every message's role, text, and image
// bytes. Because the client re-sends history verbatim each turn, the messages
// up to and including an image are fixed across turns, so this key is stable
// and later turns hit the cache. A different conversation — or a different
// question accompanying the same image — yields a different key, so the
// captioner is re-invoked with the new context.
func captionContextKey(msgs []api.Message) string {
	h := sha256.New()
	for _, m := range msgs {
		h.Write([]byte(m.Role))
		h.Write([]byte{0})
		h.Write([]byte(m.Content))
		h.Write([]byte{0})
		for _, img := range m.Images {
			h.Write([]byte(hashImage(img)))
		}
		h.Write([]byte{0})
	}
	return hex.EncodeToString(h.Sum(nil))
}

func hashImage(img api.ImageData) string {
	h := sha256.Sum256(img)
	return hex.EncodeToString(h[:])
}

// imageCaptionCache stores caption-cache-key -> caption so repeated images
// (which reappear in every turn of a stateless conversation) are captioned once
// per (image, user intent).
type imageCaptionCacheType struct {
	mu    sync.Mutex
	cache map[string]string
}

var imageCaptionCache = &imageCaptionCacheType{cache: make(map[string]string)}

const maxImageCaptionCacheEntries = 256

func (cc *imageCaptionCacheType) get(hash string) (string, bool) {
	cc.mu.Lock()
	defer cc.mu.Unlock()
	v, ok := cc.cache[hash]
	return v, ok
}

func (cc *imageCaptionCacheType) put(hash, caption string) {
	cc.mu.Lock()
	defer cc.mu.Unlock()
	if len(cc.cache) >= maxImageCaptionCacheEntries {
		// Bound memory by evicting an arbitrary entry.
		for k := range cc.cache {
			delete(cc.cache, k)
			break
		}
	}
	cc.cache[hash] = caption
}

func (cc *imageCaptionCacheType) clear() {
	cc.mu.Lock()
	defer cc.mu.Unlock()
	cc.cache = make(map[string]string)
}

// nonVisionModels records primary models that rejected image input, so later
// image-bearing turns skip the doomed first attempt and go straight to
// caption-then-primary.
type nonVisionCacheType struct {
	mu  sync.Mutex
	set map[string]struct{}
}

var nonVisionModels = &nonVisionCacheType{set: make(map[string]struct{})}

func (n *nonVisionCacheType) isNonVision(model string) bool {
	n.mu.Lock()
	defer n.mu.Unlock()
	_, ok := n.set[model]
	return ok
}

func (n *nonVisionCacheType) mark(model string) {
	n.mu.Lock()
	defer n.mu.Unlock()
	n.set[model] = struct{}{}
}

func (n *nonVisionCacheType) clear() {
	n.mu.Lock()
	defer n.mu.Unlock()
	n.set = make(map[string]struct{})
}

// cloudModelInfo holds the bits of /api/show the vision-fallback path needs:
// the model's context window (tokens) and its declared capabilities.
type cloudModelInfo struct {
	contextLength int
	capabilities  []model.Capability
}

// cloudModelInfoCache memoizes /api/show results per model so the captioner can
// size/trim the caption request and decide whether the primary can handle
// images without re-fetching per turn.
type cloudModelInfoCacheType struct {
	mu    sync.Mutex
	cache map[string]cloudModelInfo
}

var cloudModelInfoCache = &cloudModelInfoCacheType{cache: make(map[string]cloudModelInfo)}

func (cl *cloudModelInfoCacheType) get(model string) (cloudModelInfo, bool) {
	cl.mu.Lock()
	defer cl.mu.Unlock()
	v, ok := cl.cache[model]
	return v, ok
}

func (cl *cloudModelInfoCacheType) put(model string, info cloudModelInfo) {
	cl.mu.Lock()
	defer cl.mu.Unlock()
	cl.cache[model] = info
}

func (cl *cloudModelInfoCacheType) clear() {
	cl.mu.Lock()
	defer cl.mu.Unlock()
	cl.cache = make(map[string]cloudModelInfo)
}

// fetchCloudModelInfo reads /api/show for a cloud model and extracts its context
// window (model_info's <arch>.context_length) and declared capabilities. The
// second return is false when the call failed or produced no usable info, so
// callers can fall back to reactive behavior rather than acting on stale/empty
// metadata.
func fetchCloudModelInfo(c *gin.Context, model, disabledOperation string) (cloudModelInfo, bool) {
	if info, ok := cloudModelInfoCache.get(model); ok {
		return info, true
	}
	info := cloudModelInfo{}
	ok := false
	if body, err := json.Marshal(api.ShowRequest{Model: model}); err == nil {
		if resp, err := executeCloudProxyRequest(c, body, "/api/show", disabledOperation); err == nil {
			respBody, _ := io.ReadAll(io.LimitReader(resp.Body, maxDecompressedBodySize))
			resp.Body.Close()
			var show api.ShowResponse
			if json.Unmarshal(respBody, &show) == nil {
				for k, v := range show.ModelInfo {
					if !strings.HasSuffix(k, ".context_length") {
						continue
					}
					if iv, ok := v.(float64); ok && int(iv) > info.contextLength {
						info.contextLength = int(iv)
					}
				}
				info.capabilities = show.Capabilities
				// "known" means /api/show actually reported a capability list we
				// can trust; an absent/empty list (or a failed call) leaves the
				// caller on the reactive try-and-catch-400 path.
				ok = len(show.Capabilities) > 0
			}
		}
	}
	cloudModelInfoCache.put(model, info)
	return info, ok
}

// cloudModelContextLength returns the cloud model's context window size in
// tokens (0 if unknown); callers treat 0 as "unknown" and skip
// context-length-dependent logic, falling back to the reactive
// trim-on-overflow retry.
func cloudModelContextLength(c *gin.Context, model, disabledOperation string) int {
	info, _ := fetchCloudModelInfo(c, model, disabledOperation)
	return info.contextLength
}

// cloudModelCapabilities returns the cloud model's declared capabilities and
// whether /api/show reported them. When known is false the caller should fall
// back to the reactive try-and-catch-400 behavior rather than trusting an empty
// capability list.
func cloudModelCapabilities(c *gin.Context, model, disabledOperation string) ([]model.Capability, bool) {
	info, known := fetchCloudModelInfo(c, model, disabledOperation)
	return info.capabilities, known
}

// hasCapability reports whether caps contains cap.
func hasCapability(caps []model.Capability, cap model.Capability) bool {
	for _, c := range caps {
		if c == cap {
			return true
		}
	}
	return false
}

// estimateContextTokens returns a rough token-count estimate for a sequence of
// messages, used to decide whether to proactively trim the caption context
// before sending. Cloud models don't expose a tokenizer (/api/tokenize returns
// 404 for cloud), so this is a conservative byte-level heuristic: ~4 text bytes
// per token plus a flat per-image allowance. It only needs to catch clearly
// oversized contexts; the reactive trim-on-overflow retry handles any
// underestimate.
func estimateContextTokens(msgs []api.Message) int {
	const bytesPerToken = 4
	const tokensPerImage = 1024
	n := 0
	for _, m := range msgs {
		n += len(m.Content) / bytesPerToken
		n += len(m.Images) * tokensPerImage
	}
	return n
}

func replaceJSONModelField(body []byte, model string) ([]byte, error) {
	if len(body) == 0 {
		return body, nil
	}

	var payload map[string]json.RawMessage
	if err := json.Unmarshal(body, &payload); err != nil {
		return nil, err
	}

	modelJSON, err := json.Marshal(model)
	if err != nil {
		return nil, err
	}
	payload["model"] = modelJSON

	return json.Marshal(payload)
}

func readRequestBody(r *http.Request) ([]byte, error) {
	if r.Body == nil {
		return nil, nil
	}

	body, err := io.ReadAll(r.Body)
	if err != nil {
		return nil, err
	}

	r.Body = io.NopCloser(bytes.NewReader(body))
	return body, nil
}

func extractModelField(body []byte) (string, bool) {
	if len(body) == 0 {
		return "", false
	}

	var payload map[string]json.RawMessage
	if err := json.Unmarshal(body, &payload); err != nil {
		return "", false
	}

	raw, ok := payload["model"]
	if !ok {
		return "", false
	}

	var model string
	if err := json.Unmarshal(raw, &model); err != nil {
		return "", false
	}

	model = strings.TrimSpace(model)
	return model, model != ""
}

func hasAnthropicWebSearchTool(body []byte) bool {
	if len(body) == 0 {
		return false
	}

	var payload struct {
		Tools []struct {
			Type string `json:"type"`
		} `json:"tools"`
	}
	if err := json.Unmarshal(body, &payload); err != nil {
		return false
	}

	for _, tool := range payload.Tools {
		if strings.HasPrefix(strings.TrimSpace(tool.Type), "web_search") {
			return true
		}
	}

	return false
}

// hasAnthropicImageContent reports whether an Anthropic /v1/messages request
// body carries any image content blocks (base64 or URL sourced). It inspects
// both top-level message content and tool_result content blocks, since
// clients such as Claude Code may return images from tool calls.
func hasAnthropicImageContent(body []byte) bool {
	if len(body) == 0 {
		return false
	}

	// A content block is "image-like" if its type is "image", or if it is a
	// tool_result whose content array contains an "image" block. Anthropic
	// message content may be either a string or an array of blocks, so the
	// content fields are decoded as raw JSON and parsed per-message; a string
	// content (no image) must not abort detection for the whole request.
	type contentBlock struct {
		Type   string `json:"type"`
		Source *struct {
			Type string `json:"type"`
		} `json:"source"`
		Content json.RawMessage `json:"content"`
	}
	var payload struct {
		Messages []struct {
			Content json.RawMessage `json:"content"`
		} `json:"messages"`
	}
	if err := json.Unmarshal(body, &payload); err != nil {
		return false
	}

	var hasImage func(raw json.RawMessage) bool
	hasImage = func(raw json.RawMessage) bool {
		var blocks []contentBlock
		if json.Unmarshal(raw, &blocks) != nil {
			return false // string (or non-array) content carries no image
		}
		for _, block := range blocks {
			switch strings.TrimSpace(block.Type) {
			case "image":
				return true
			case "tool_result":
				if hasImage(block.Content) {
					return true
				}
			}
		}
		return false
	}

	for _, msg := range payload.Messages {
		if hasImage(msg.Content) {
			return true
		}
	}
	return false
}

// isImageNotSupportedError reports whether a cloud error response body
// indicates the requested model does not support image input. The remote
// returns this as a 400 with a JSON {"error": "..."} body; we match a few
// phrasings so the fallback is robust to wording changes.
func isImageNotSupportedError(body []byte) bool {
	if len(body) == 0 {
		return false
	}

	var errData struct {
		Error string `json:"error"`
	}
	if err := json.Unmarshal(body, &errData); err != nil {
		// Fall back to matching against the raw body.
		errData.Error = string(body)
	}

	msg := strings.ToLower(errData.Error)
	switch {
	case strings.Contains(msg, "does not support image input"),
		strings.Contains(msg, "does not support images"),
		strings.Contains(msg, "not support image"),
		strings.Contains(msg, "image input not supported"),
		strings.Contains(msg, "image input is not supported"):
		return true
	}
	return false
}

func writeCloudUnauthorized(c *gin.Context) {
	signinURL, err := cloudProxySigninURL()
	if err != nil {
		c.JSON(http.StatusUnauthorized, gin.H{"error": "unauthorized"})
		return
	}

	c.JSON(http.StatusUnauthorized, gin.H{"error": "unauthorized", "signin_url": signinURL})
}

func signCloudProxyRequest(ctx context.Context, req *http.Request) error {
	if !strings.EqualFold(req.URL.Hostname(), cloudProxySigningHost) {
		return nil
	}

	ts := strconv.FormatInt(time.Now().Unix(), 10)
	challenge := buildCloudSignatureChallenge(req, ts)
	signature, err := auth.Sign(ctx, []byte(challenge))
	if err != nil {
		return err
	}

	req.Header.Set("Authorization", signature)
	return nil
}

func buildCloudSignatureChallenge(req *http.Request, ts string) string {
	query := req.URL.Query()
	query.Set("ts", ts)
	req.URL.RawQuery = query.Encode()

	return fmt.Sprintf("%s,%s", req.Method, req.URL.RequestURI())
}

func resolveCloudProxyBaseURL(rawOverride string, runMode string) (baseURL string, signingHost string, overridden bool, err error) {
	baseURL = defaultCloudProxyBaseURL
	signingHost = defaultCloudProxySigningHost

	rawOverride = strings.TrimSpace(rawOverride)
	if rawOverride == "" {
		return baseURL, signingHost, false, nil
	}

	u, err := url.Parse(rawOverride)
	if err != nil {
		return "", "", false, fmt.Errorf("invalid URL: %w", err)
	}
	if u.Scheme == "" || u.Host == "" {
		return "", "", false, fmt.Errorf("invalid URL: scheme and host are required")
	}
	if u.User != nil {
		return "", "", false, fmt.Errorf("invalid URL: userinfo is not allowed")
	}
	if u.Path != "" && u.Path != "/" {
		return "", "", false, fmt.Errorf("invalid URL: path is not allowed")
	}
	if u.RawQuery != "" || u.Fragment != "" {
		return "", "", false, fmt.Errorf("invalid URL: query and fragment are not allowed")
	}

	host := u.Hostname()
	if host == "" {
		return "", "", false, fmt.Errorf("invalid URL: host is required")
	}

	loopback := isLoopbackHost(host)
	if runMode == gin.ReleaseMode && !loopback {
		return "", "", false, fmt.Errorf("non-loopback cloud override is not allowed in release mode")
	}
	if !loopback && !strings.EqualFold(u.Scheme, "https") {
		return "", "", false, fmt.Errorf("non-loopback cloud override must use https")
	}

	u.Path = ""
	u.RawPath = ""
	u.RawQuery = ""
	u.Fragment = ""

	return u.String(), strings.ToLower(host), true, nil
}

func isLoopbackHost(host string) bool {
	if strings.EqualFold(host, "localhost") {
		return true
	}

	ip := net.ParseIP(host)
	return ip != nil && ip.IsLoopback()
}

func copyProxyRequestHeaders(dst, src http.Header) {
	connectionTokens := connectionHeaderTokens(src)
	for key, values := range src {
		if isHopByHopHeader(key) || isConnectionTokenHeader(key, connectionTokens) {
			continue
		}

		dst.Del(key)
		for _, value := range values {
			dst.Add(key, value)
		}
	}
}

func copyProxyResponseHeaders(dst, src http.Header) {
	connectionTokens := connectionHeaderTokens(src)
	for key, values := range src {
		if isHopByHopHeader(key) || isConnectionTokenHeader(key, connectionTokens) {
			continue
		}

		dst.Del(key)
		for _, value := range values {
			dst.Add(key, value)
		}
	}
}

func copyProxyResponseBody(dst http.ResponseWriter, src io.Reader) error {
	flusher, canFlush := dst.(http.Flusher)
	buf := make([]byte, 32*1024)

	for {
		n, err := src.Read(buf)
		if n > 0 {
			if _, writeErr := dst.Write(buf[:n]); writeErr != nil {
				return writeErr
			}
			if canFlush {
				// TODO(drifkin): Consider conditional flushing so non-streaming
				// responses don't flush every write and can optimize throughput.
				flusher.Flush()
			}
		}

		if err != nil {
			if err == io.EOF {
				return nil
			}
			return err
		}
	}
}

type jsonlFramingResponseWriter struct {
	http.ResponseWriter
	pending []byte
}

func (w *jsonlFramingResponseWriter) Flush() {
	if flusher, ok := w.ResponseWriter.(http.Flusher); ok {
		flusher.Flush()
	}
}

func (w *jsonlFramingResponseWriter) Write(p []byte) (int, error) {
	w.pending = append(w.pending, p...)
	if err := w.flushCompleteLines(); err != nil {
		return len(p), err
	}
	return len(p), nil
}

func (w *jsonlFramingResponseWriter) FlushPending() error {
	trailing := bytes.TrimSpace(w.pending)
	w.pending = nil
	if len(trailing) == 0 {
		return nil
	}

	_, err := w.ResponseWriter.Write(trailing)
	return err
}

func (w *jsonlFramingResponseWriter) flushCompleteLines() error {
	for {
		newline := bytes.IndexByte(w.pending, '\n')
		if newline < 0 {
			return nil
		}

		line := bytes.TrimSpace(w.pending[:newline])
		w.pending = w.pending[newline+1:]
		if len(line) == 0 {
			continue
		}

		if _, err := w.ResponseWriter.Write(line); err != nil {
			return err
		}
	}
}

func isHopByHopHeader(name string) bool {
	_, ok := hopByHopHeaders[strings.ToLower(name)]
	return ok
}

func connectionHeaderTokens(header http.Header) map[string]struct{} {
	tokens := map[string]struct{}{}
	for _, raw := range header.Values("Connection") {
		for _, token := range strings.Split(raw, ",") {
			token = strings.TrimSpace(strings.ToLower(token))
			if token == "" {
				continue
			}
			tokens[token] = struct{}{}
		}
	}
	return tokens
}

func isConnectionTokenHeader(name string, tokens map[string]struct{}) bool {
	if len(tokens) == 0 {
		return false
	}
	_, ok := tokens[strings.ToLower(name)]
	return ok
}
