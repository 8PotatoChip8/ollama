package server

import (
	"bytes"
	"context"
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
	"time"

	"github.com/gin-gonic/gin"
	"github.com/klauspost/compress/zstd"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/auth"
	"github.com/ollama/ollama/envconfig"
	internalcloud "github.com/ollama/ollama/internal/cloud"
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
		return
	}
	defer resp.Body.Close()
	streamCloudResponse(c, resp, path)
}

// executeCloudProxyRequest builds, signs, and sends a proxied request to the
// cloud at the given path. The caller is responsible for closing resp.Body.
// On a setup failure (cloud disabled, bad base URL, signing failure, transport
// error) it writes an error response to c and returns a non-nil err.
func executeCloudProxyRequest(c *gin.Context, body []byte, path, disabledOperation string) (*http.Response, error) {
	if disabled, _ := internalcloud.Status(); disabled {
		c.JSON(http.StatusForbidden, gin.H{"error": internalcloud.DisabledError(disabledOperation)})
		return nil, errors.New(internalcloud.DisabledError(disabledOperation))
	}

	baseURL, err := url.Parse(cloudProxyBaseURL)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return nil, err
	}

	targetURL := baseURL.ResolveReference(&url.URL{
		Path:     path,
		RawQuery: c.Request.URL.RawQuery,
	})

	outReq, err := http.NewRequestWithContext(c.Request.Context(), c.Request.Method, targetURL.String(), bytes.NewReader(body))
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return nil, err
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
		writeCloudUnauthorized(c)
		return nil, err
	}

	// TODO(drifkin): Add phase-specific proxy timeouts.
	// Connect/TLS/TTFB should have bounded timeouts, but once streaming starts
	// we should not enforce a short total timeout for long-lived responses.
	resp, err := http.DefaultClient.Do(outReq)
	if err != nil {
		c.JSON(http.StatusBadGateway, gin.H{"error": err.Error()})
		return nil, err
	}
	return resp, nil
}

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
// to the cloud for a diverted Anthropic /v1/messages image request. If the
// cloud rejects the request because the model does not support image input and
// OLLAMA_CLOUD_VISION_FALLBACK is configured, it retries the request with the
// fallback model and streams that response instead. With no fallback
// configured it behaves exactly like the normal converted cloud path.
func proxyCloudChatWithVisionFallback(c *gin.Context, req api.ChatRequest, disabledOperation string) {
	fallback := strings.TrimSpace(envconfig.CloudVisionFallback())
	if fallback == "" {
		proxyCloudJSONRequestWithPath(c, req, "/api/chat", disabledOperation)
		return
	}

	fallbackRef, err := parseAndValidateModelRef(fallback)
	if err != nil {
		slog.Warn("invalid OLLAMA_CLOUD_VISION_FALLBACK model, ignoring", "value", fallback, "error", err)
		proxyCloudJSONRequestWithPath(c, req, "/api/chat", disabledOperation)
		return
	}

	body, err := json.Marshal(req)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	resp, err := executeCloudProxyRequest(c, body, "/api/chat", disabledOperation)
	if err != nil {
		return
	}

	// A 400 may be the "model does not support image input" rejection we want
	// to fall back from. Buffer it so we can inspect it and, if needed, retry
	// without having committed any response bytes to the client.
	if resp.StatusCode == http.StatusBadRequest {
		errBody, readErr := io.ReadAll(io.LimitReader(resp.Body, maxDecompressedBodySize))
		resp.Body.Close()
		if readErr != nil {
			c.JSON(http.StatusBadGateway, gin.H{"error": readErr.Error()})
			return
		}

		if isImageNotSupportedError(errBody) {
			slog.Info("cloud model does not support image input, retrying with vision fallback",
				"model", req.Model, "fallback", fallbackRef.Base)
			req.Model = fallbackRef.Base
			body2, err := json.Marshal(req)
			if err != nil {
				c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
				return
			}
			resp2, err := executeCloudProxyRequest(c, body2, "/api/chat", disabledOperation)
			if err != nil {
				return
			}
			defer resp2.Body.Close()
			streamCloudResponse(c, resp2, "/api/chat")
			return
		}

		// Not an image rejection: surface the original error. The downstream
		// AnthropicWriter translates non-200 bodies into Anthropic-format errors.
		copyProxyResponseHeaders(c.Writer.Header(), resp.Header)
		c.Status(http.StatusBadRequest)
		if _, err := c.Writer.Write(errBody); err != nil {
			slog.Warn("cloud proxy error body write failed", "error", err)
		}
		return
	}

	defer resp.Body.Close()
	streamCloudResponse(c, resp, "/api/chat")
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
	// tool_result whose content array contains an "image" block.
	type contentBlock struct {
		Type   string `json:"type"`
		Source *struct {
			Type string `json:"type"`
		} `json:"source"`
		Content []contentBlock `json:"content"`
	}
	var payload struct {
		Messages []struct {
			Content []contentBlock `json:"content"`
		} `json:"messages"`
	}
	if err := json.Unmarshal(body, &payload); err != nil {
		return false
	}

	var hasImage func(blocks []contentBlock) bool
	hasImage = func(blocks []contentBlock) bool {
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
