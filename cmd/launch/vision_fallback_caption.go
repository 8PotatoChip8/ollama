package launch

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"sync"

	"github.com/ollama/ollama/anthropic"
	"github.com/ollama/ollama/api"
	modelpkg "github.com/ollama/ollama/types/model"
)

// maxCaptionTokens caps the fallback's response length when captioning, so a
// caption can't grow to the primary's (potentially very large) max_tokens.
const maxCaptionTokens = 2048

// maxCaptionBodySize bounds how much of a caption /api/show response we buffer.
const maxCaptionBodySize = 20 << 20

// captionMessageInContext asks the fallback vision model to handle the
// image(s) in req.Messages[msgIdx] by sending it the same request the primary
// would have received up to that point — system prompt, prior turns, and this
// message (text + image) — with the model swapped to the fallback and
// tools/thinking stripped. The fallback sees the image with the full task
// context, so its response reflects the conversation, not a context-free
// "describe this". That response becomes the caption the primary reasons over.
//
// Only messages up to and including msgIdx are sent: the Messages API is
// stateless and the client re-sends history verbatim each turn, so everything
// up to the image is fixed across turns while later turns grow. Trimming here
// keeps the caption cache stable so later turns hit the cache (single primary
// call). The captioner issues its own /v1/messages request to the server and
// must not write to the client response.
func captionMessageInContext(client *http.Client, baseURL string, req anthropic.MessagesRequest, msgIdx int, fallback string) (string, error) {
	contextMsgs := req.Messages[:msgIdx+1]
	key := captionContextKey(contextMsgs)
	if caption, ok := imageCaptionCache.get(key); ok {
		return caption, nil
	}

	msgs := contextMsgs
	// Proactively trim the conversation to the fallback's context window before
	// sending, so we don't rely on the cloud's default/truncation behavior. The
	// image-bearing message is the last in contextMsgs, so it is never dropped;
	// the system prompt (if present as a message) is also kept. Cloud models
	// don't expose a tokenizer, so this uses a conservative estimate; the
	// reactive trim-on-overflow retry below handles any underestimate.
	if ctxLen := cloudModelContextLength(client, baseURL, fallback); ctxLen > 0 {
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
		capReq := buildCaptionRequest(req, msgs, fallback)
		body, err := encodeMessagesRequest(capReq)
		if err != nil {
			return "", err
		}
		resp, err := postToServer(client, baseURL, "/v1/messages", body)
		if err != nil {
			return "", err
		}
		respBody, rerr := io.ReadAll(io.LimitReader(resp.Body, maxCaptionBodySize))
		resp.Body.Close()
		if rerr != nil {
			return "", rerr
		}
		if resp.StatusCode == http.StatusOK {
			var mr anthropic.MessagesResponse
			if err := json.Unmarshal(respBody, &mr); err != nil {
				return "", fmt.Errorf("decode caption response: %w", err)
			}
			caption = extractAssistantText(mr)
			if caption == "" {
				return "", fmt.Errorf("caption model %q returned empty response", fallback)
			}
			break
		}
		if !isContextLengthError(respBody) {
			return "", fmt.Errorf("caption model %q returned status %d: %s", fallback, resp.StatusCode, strings.TrimSpace(string(respBody)))
		}
		next, ok := dropOldestNonSystem(msgs)
		if !ok {
			return "", fmt.Errorf("caption model %q context too small even for system + image message", fallback)
		}
		msgs = next
	}

	imageCaptionCache.put(key, caption)
	return caption, nil
}

// captionRequestImages replaces every image in req with a text caption
// produced by the fallback vision model, captioning each image-bearing message
// in the full conversation context it appeared in. It captions all images
// first and only mutates req once every caption succeeds, so a failure leaves
// req unchanged. Returns false (with req unmodified) if any image could not be
// captioned.
func captionRequestImages(client *http.Client, baseURL string, req *anthropic.MessagesRequest, fallback string) bool {
	type msgCaption struct {
		idx  int
		text string
	}
	var captions []msgCaption
	for i := range req.Messages {
		if !messageHasImage(req.Messages[i]) {
			continue
		}
		caption, err := captionMessageInContext(client, baseURL, *req, i, fallback)
		if err != nil {
			return false
		}
		captions = append(captions, msgCaption{i, caption})
	}

	for _, cap := range captions {
		replaceImagesInMessage(&req.Messages[cap.idx], captionText(fallback, cap.text))
	}
	return true
}

// isImageNotSupportedError reports whether a server error response body
// indicates the requested model does not support image input. The server
// returns this as a 400; we match a few phrasings so the fallback is robust to
// wording changes.
func isImageNotSupportedError(body []byte) bool {
	if len(body) == 0 {
		return false
	}
	var errData struct {
		Error any `json:"error"`
	}
	msg := ""
	if err := json.Unmarshal(body, &errData); err == nil {
		msg = errorString(errData.Error)
	} else {
		msg = string(body)
	}
	msg = strings.ToLower(msg)
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

// errorString flattens an Anthropic-style error value (which may be a string
// or an object with a "message" field) into a single message string.
func errorString(e any) string {
	switch v := e.(type) {
	case nil:
		return ""
	case string:
		return v
	case []any:
		var b strings.Builder
		for _, item := range v {
			b.WriteString(errorString(item))
		}
		return b.String()
	case map[string]any:
		if m, ok := v["message"].(string); ok {
			return m
		}
		// Fall back to a compact rendering so something is matched.
		raw, _ := json.Marshal(v)
		return string(raw)
	default:
		return fmt.Sprintf("%v", v)
	}
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
// preserving a leading system message (if any) and the trailing image-bearing
// message. It returns ok=false when no message can be dropped without removing
// the system prompt or the final (image) message.
func dropOldestNonSystem(msgs []anthropic.MessageParam) ([]anthropic.MessageParam, bool) {
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
	out := make([]anthropic.MessageParam, 0, len(msgs)-1)
	out = append(out, msgs[:start]...)
	out = append(out, msgs[start+1:]...)
	return out, true
}

// captionContextKey returns a stable cache key for captioning the image(s) in
// the last message of msgs: a hash over every message's role, block types,
// text, and image bytes. Because the client re-sends history verbatim each
// turn, the messages up to and including an image are fixed across turns, so
// this key is stable and later turns hit the cache. A different conversation —
// or a different question accompanying the same image — yields a different
// key, so the captioner is re-invoked with the new context.
func captionContextKey(msgs []anthropic.MessageParam) string {
	h := sha256.New()
	for _, m := range msgs {
		h.Write([]byte(m.Role))
		h.Write([]byte{0})
		for _, b := range m.Content {
			h.Write([]byte(b.Type))
			h.Write([]byte{0})
			if b.Text != nil {
				h.Write([]byte(*b.Text))
			}
			h.Write([]byte{0})
			if b.Source != nil {
				h.Write([]byte(hashImageSource(*b.Source)))
			}
			h.Write([]byte{0})
			if b.Type == "tool_result" {
				if blocks, ok := toolResultContentBlocks(b.Content); ok {
					for _, cb := range blocks {
						if cb.Source != nil {
							h.Write([]byte(hashImageSource(*cb.Source)))
						}
					}
				}
			}
		}
		h.Write([]byte{0})
	}
	return hex.EncodeToString(h.Sum(nil))
}

func hashImageSource(src anthropic.ImageSource) string {
	h := sha256.Sum256([]byte(src.Data))
	return hex.EncodeToString(h[:])
}

// estimateContextTokens returns a rough token-count estimate for a sequence of
// messages, used to decide whether to proactively trim the caption context
// before sending. Cloud models don't expose a tokenizer, so this is a
// conservative byte-level heuristic: ~4 text bytes per token plus a flat
// per-image allowance. It only needs to catch clearly oversized contexts; the
// reactive trim-on-overflow retry handles any underestimate.
func estimateContextTokens(msgs []anthropic.MessageParam) int {
	const bytesPerToken = 4
	const tokensPerImage = 1024
	n := 0
	for _, m := range msgs {
		for _, b := range m.Content {
			switch b.Type {
			case "text":
				if b.Text != nil {
					n += len(*b.Text) / bytesPerToken
				}
			case "image":
				n += tokensPerImage
			case "tool_result":
				n += estimateToolResultTokens(b.Content)
			}
		}
	}
	return n
}

func estimateToolResultTokens(c any) int {
	blocks, ok := toolResultContentBlocks(c)
	if !ok {
		return 0
	}
	const bytesPerToken = 4
	const tokensPerImage = 1024
	n := 0
	for _, b := range blocks {
		switch b.Type {
		case "text":
			if b.Text != nil {
				n += len(*b.Text) / bytesPerToken
			}
		case "image":
			n += tokensPerImage
		}
	}
	return n
}

// postToServer sends a JSON POST to the Ollama server at baseURL+path. The
// server signs cloud requests itself, so the client-side proxy needs no cloud
// credentials — it is a plain HTTP client of the local daemon.
func postToServer(client *http.Client, baseURL, path string, body []byte) (*http.Response, error) {
	httpReq, err := http.NewRequest(http.MethodPost, baseURL+path, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	httpReq.Header.Set("Content-Type", "application/json")
	return client.Do(httpReq)
}

// imageCaptionCache stores caption-cache-key -> caption so repeated images
// (which reappear in every turn of a stateless conversation) are captioned
// once per (image, user intent).
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

// nonVisionCache records primary models that rejected image input, so later
// image-bearing turns skip the doomed first attempt and go straight to
// caption-then-primary.
type nonVisionCacheType struct {
	mu  sync.Mutex
	set map[string]struct{}
}

var nonVisionCache = &nonVisionCacheType{set: make(map[string]struct{})}

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
	capabilities  []modelpkg.Capability
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

// resetVisionFallbackCaches clears all vision-fallback caches. Tests call this
// between cases so cached /api/show lookups and captions from one case do not
// leak into another.
func resetVisionFallbackCaches() {
	imageCaptionCache.clear()
	nonVisionCache.clear()
	cloudModelInfoCache.clear()
}

// fetchCloudModelInfo reads /api/show for a cloud model (the local server
// proxies /api/show to the cloud for cloud models) and extracts its context
// window (model_info's <arch>.context_length) and declared capabilities. The
// second return is false when the call failed or produced no capability list,
// so callers fall back to the reactive try-and-catch-400 path rather than
// acting on stale/empty metadata.
func fetchCloudModelInfo(client *http.Client, baseURL, model string) (cloudModelInfo, bool) {
	if info, ok := cloudModelInfoCache.get(model); ok {
		return info, true
	}
	info := cloudModelInfo{}
	known := false
	body, err := json.Marshal(api.ShowRequest{Model: model, Name: model})
	if err == nil {
		resp, err := postToServer(client, baseURL, "/api/show", body)
		if err == nil {
			respBody, _ := io.ReadAll(io.LimitReader(resp.Body, maxCaptionBodySize))
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
				// "known" means /api/show reported a capability list we can
				// trust; an absent/empty list (or a failed call) leaves the
				// caller on the reactive try-and-catch-400 path.
				known = len(show.Capabilities) > 0
			}
		}
	}
	cloudModelInfoCache.put(model, info)
	return info, known
}

func cloudModelContextLength(client *http.Client, baseURL, model string) int {
	info, _ := fetchCloudModelInfo(client, baseURL, model)
	return info.contextLength
}

// cloudModelCapabilities returns the cloud model's declared capabilities and
// whether /api/show reported them. When known is false the caller should fall
// back to the reactive try-and-catch-400 behavior rather than trusting an
// empty capability list.
func cloudModelCapabilities(client *http.Client, baseURL, model string) ([]modelpkg.Capability, bool) {
	info, known := fetchCloudModelInfo(client, baseURL, model)
	return info.capabilities, known
}

// hasCapability reports whether caps contains cap.
func hasCapability(caps []modelpkg.Capability, cap modelpkg.Capability) bool {
	for _, c := range caps {
		if c == cap {
			return true
		}
	}
	return false
}
