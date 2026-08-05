package launch

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net"
	"net/http"
	"net/http/httputil"
	"net/url"
	"strings"

	"github.com/ollama/ollama/anthropic"
	"github.com/ollama/ollama/envconfig"
	modelpkg "github.com/ollama/ollama/types/model"
)

// startVisionFallbackProxy starts a localhost reverse proxy for one launch
// that intercepts POST /v1/messages. Requests without image content pass
// through to the Ollama server untouched (native Anthropic passthrough,
// streamed with full fidelity). Image-bearing requests to a non-vision primary
// are captioned by the fallback vision model and re-sent to the primary as
// text; image-bearing requests to a vision-capable primary pass through
// natively. It returns the proxy URL to point an integration at and a stop
// function that closes the server.
//
// Each launch is its own process with its own proxy on its own port, so
// concurrent launches get isolated fallbacks with no cross-talk. The proxy is
// a plain HTTP client of the local Ollama server — the server signs cloud
// requests itself, so the proxy needs no cloud credentials.
func startVisionFallbackProxy(primary, fallback string) (string, func() error, error) {
	target := envconfig.ConnectableHost()
	client := &http.Client{} // no overall timeout: streaming primary responses are long-lived

	passthrough := httputil.NewSingleHostReverseProxy(target)
	originalDirector := passthrough.Director
	passthrough.Director = func(req *http.Request) {
		originalDirector(req)
		req.Host = target.Host
	}
	passthrough.FlushInterval = -1 // unbuffered SSE streaming
	passthrough.ErrorHandler = proxyErrorHandler

	handler := &visionFallbackHandler{
		primary:     primary,
		fallback:    fallback,
		target:      target,
		client:      client,
		passthrough: passthrough,
	}

	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		return "", nil, fmt.Errorf("start vision fallback proxy: %w", err)
	}
	server := &http.Server{Handler: handler}
	go func() {
		if err := server.Serve(listener); err != nil && err != http.ErrServerClosed {
			slog.Warn("vision fallback proxy stopped", "error", err)
		}
	}()

	proxyURL := "http://" + listener.Addr().String()
	stop := func() error { return server.Close() }
	return proxyURL, stop, nil
}

// proxyErrorHandler is the reverse proxy's ErrorHandler. A client disconnect
// mid-stream (the user stops generation, the agent moves on, or the request is
// canceled) surfaces here as context.Canceled: the client is already gone, so
// logging it at Error and writing a 502 is noise. Log it at Debug and return.
// Any other error is a real transport failure worth surfacing as a 502.
func proxyErrorHandler(w http.ResponseWriter, r *http.Request, err error) {
	if errors.Is(err, context.Canceled) {
		slog.Debug("vision fallback proxy: client canceled request", "path", r.URL.Path)
		return
	}
	slog.Error("vision fallback proxy error", "error", err, "path", r.URL.Path)
	http.Error(w, "proxy error: "+err.Error(), http.StatusBadGateway)
}

// visionFallbackHandler intercepts POST /v1/messages and routes image-bearing
// requests through caption-then-primary when the primary cannot see images.
// Every other request (and every non-image /v1/messages request) passes
// through the reverse proxy untouched.
type visionFallbackHandler struct {
	primary     string
	fallback    string
	target      *url.URL
	client      *http.Client
	passthrough *httputil.ReverseProxy
}

func (h *visionFallbackHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	// Only /v1/messages needs inspection; everything else passes through.
	if r.Method != http.MethodPost || r.URL.Path != "/v1/messages" {
		h.passthrough.ServeHTTP(w, r)
		return
	}

	body, err := io.ReadAll(r.Body)
	if err != nil {
		http.Error(w, "read request: "+err.Error(), http.StatusBadRequest)
		return
	}
	r.Body.Close()

	req, err := parseMessagesRequest(body)
	if err != nil || !hasImageContent(req) {
		// Not a valid Messages request, or no image to handle: pass the
		// original body through untouched.
		h.forwardRaw(w, r, body)
		return
	}

	h.handleVisionFallback(w, r, body, req)
}

// handleVisionFallback runs the caption-then-primary flow for an
// image-bearing request. A vision-capable primary (or an unknown one that
// accepts the image) is passed the real image bytes; a non-vision primary gets
// the image replaced with a fallback-generated caption before being sent.
func (h *visionFallbackHandler) handleVisionFallback(w http.ResponseWriter, r *http.Request, body []byte, req anthropic.MessagesRequest) {
	baseURL := h.target.String()

	// Decide if the primary is non-vision: proactively via /api/show
	// capabilities (avoids a wasted doomed attempt on the first image turn),
	// or from a prior rejection. When /api/show is silent we fall back to
	// trying the primary and catching the 400 below.
	primaryNonVision := nonVisionCache.isNonVision(req.Model)
	if !primaryNonVision {
		if caps, known := cloudModelCapabilities(h.client, baseURL, req.Model); known && !hasCapability(caps, modelpkg.CapabilityVision) {
			primaryNonVision = true
			nonVisionCache.mark(req.Model)
		}
	}

	if !primaryNonVision {
		// Try the primary with the real image bytes. On 200 (or any non-400),
		// stream through. On a 400 image-not-supported, mark non-vision and
		// fall through to caption-then-primary.
		if h.tryPrimaryWithImage(w, r, body, req.Model) {
			return
		}
	}

	// Caption every image (cached), replace image blocks with caption text,
	// and re-send to the primary. On caption failure, degrade to letting the
	// fallback model answer this turn directly so the user still gets a reply.
	if !captionRequestImages(h.client, baseURL, &req, h.fallback) {
		slog.Warn("image captioning failed, degrading to vision fallback model for this turn", "fallback", h.fallback)
		req.Model = h.fallback
	}

	modified, err := encodeMessagesRequest(req)
	if err != nil {
		http.Error(w, "encode fallback request: "+err.Error(), http.StatusInternalServerError)
		return
	}
	h.forwardRaw(w, r, modified)
}

// tryPrimaryWithImage forwards the original image-bearing request to the
// primary. It returns true if it fully handled the response (streamed it
// through, or surfaced a non-image 400, or hit a transport error). It returns
// false only when the primary rejected the image with a 400
// image-not-supported, so the caller can fall through to caption-then-primary.
func (h *visionFallbackHandler) tryPrimaryWithImage(w http.ResponseWriter, r *http.Request, body []byte, model string) bool {
	resp, err := h.doServerRequest(http.MethodPost, "/v1/messages", body, r.Header)
	if err != nil {
		http.Error(w, "proxy error: "+err.Error(), http.StatusBadGateway)
		return true
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusBadRequest {
		h.streamResponse(w, resp)
		return true
	}

	// 400: buffer to inspect. If it's not an image-not-supported error,
	// surface it as-is; otherwise fall through to captioning.
	errBody, readErr := io.ReadAll(io.LimitReader(resp.Body, maxCaptionBodySize))
	if readErr != nil {
		http.Error(w, "read primary error: "+readErr.Error(), http.StatusBadGateway)
		return true
	}
	if !isImageNotSupportedError(errBody) {
		copyResponseHeaders(w.Header(), resp.Header)
		w.WriteHeader(http.StatusBadRequest)
		_, _ = w.Write(errBody)
		return true
	}

	nonVisionCache.mark(model)
	slog.Info("cloud model does not support image input, captioning via fallback then retrying primary",
		"model", model, "fallback", h.fallback)
	return false
}

// forwardRaw forwards a raw body to the server via the reverse proxy, which
// streams the response back to the client with full fidelity. It is used for
// pass-through (no image, or non-/v1/messages) and for the captioned primary
// request.
func (h *visionFallbackHandler) forwardRaw(w http.ResponseWriter, r *http.Request, body []byte) {
	r.Body = io.NopCloser(bytes.NewReader(body))
	r.ContentLength = int64(len(body))
	h.passthrough.ServeHTTP(w, r)
}

// streamResponse copies a sub-request response to the client, flushing as it
// goes so SSE streams reach the client promptly. Used by the reactive primary
// attempt, which cannot use the reverse proxy because it must inspect a 400
// before committing bytes to the client.
func (h *visionFallbackHandler) streamResponse(w http.ResponseWriter, resp *http.Response) {
	copyResponseHeaders(w.Header(), resp.Header)
	w.WriteHeader(resp.StatusCode)
	flusher, _ := w.(http.Flusher)
	buf := make([]byte, 32*1024)
	for {
		n, err := resp.Body.Read(buf)
		if n > 0 {
			_, _ = w.Write(buf[:n])
			if flusher != nil {
				flusher.Flush()
			}
		}
		if err != nil {
			return
		}
	}
}

// doServerRequest sends a request to the Ollama server at baseURL+path,
// forwarding the client's headers (minus hop-by-hop and length/host) so any
// Anthropic-version or accept headers the integration sent reach the server.
func (h *visionFallbackHandler) doServerRequest(method, path string, body []byte, srcHeaders http.Header) (*http.Response, error) {
	httpReq, err := http.NewRequest(method, h.target.String()+path, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	httpReq.Header.Set("Content-Type", "application/json")
	for k, vv := range srcHeaders {
		if isRequestHopByHop(k) {
			continue
		}
		for _, v := range vv {
			httpReq.Header.Add(k, v)
		}
	}
	return h.client.Do(httpReq)
}

// responseHopByHop are response headers the HTTP transport manages itself, so
// the proxy must not copy them when streaming a sub-request response.
var responseHopByHop = map[string]struct{}{
	"connection":          {},
	"keep-alive":          {},
	"proxy-authenticate":  {},
	"proxy-authorization": {},
	"te":                  {},
	"trailers":            {},
	"transfer-encoding":   {},
	"upgrade":             {},
	"content-length":      {}, // recomputed by the response writer
}

func copyResponseHeaders(dst, src http.Header) {
	for k, vv := range src {
		if _, skip := responseHopByHop[strings.ToLower(k)]; skip {
			continue
		}
		for _, v := range vv {
			dst.Add(k, v)
		}
	}
}

// requestHopByHop are request headers that are hop-by-hop and must not be
// forwarded by a proxy, plus content-length/host which are recomputed for the
// sub-request.
var requestHopByHop = map[string]struct{}{
	"connection":          {},
	"keep-alive":          {},
	"proxy-authenticate":  {},
	"proxy-authorization": {},
	"te":                  {},
	"trailers":            {},
	"transfer-encoding":   {},
	"upgrade":             {},
	"content-length":      {},
	"host":                {},
}

func isRequestHopByHop(name string) bool {
	_, ok := requestHopByHop[strings.ToLower(name)]
	return ok
}
