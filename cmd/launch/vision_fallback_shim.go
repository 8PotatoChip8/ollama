package launch

import (
	"context"
	"net"
	"net/http"
	"net/http/httputil"
	"time"

	"github.com/ollama/ollama/envconfig"
)

// visionFallbackHeader is the per-request header a launch shim injects so the
// server uses this launch's fallback instead of the server-wide
// OLLAMA_CLOUD_VISION_FALLBACK env var. See server/cloud_proxy.go.
const visionFallbackHeader = "X-Ollama-Cloud-Vision-Fallback"

// startVisionFallbackShim starts a localhost reverse proxy that forwards to the
// Ollama server (envconfig.ConnectableHost) and injects X-Ollama-Cloud-Vision-Fallback
// on every request, so a per-launch cloud vision fallback can be selected without a
// server-wide setting. It mirrors the desktop app's daemon reverse proxy
// (app/ui/ui.go). Returns the shim's base URL (for ANTHROPIC_BASE_URL) and a stop
// function that closes the listener and drains in-flight requests.
//
// Each launch runs its own shim on its own port, so concurrent launches with
// different --fallback values are isolated.
func startVisionFallbackShim(fallback string) (string, func(), error) {
	target := envconfig.ConnectableHost()

	proxy := httputil.NewSingleHostReverseProxy(target)
	// Flush immediately so SSE streams (Claude Code streams /v1/messages) are
	// forwarded incrementally instead of being buffered.
	proxy.FlushInterval = -1

	originalDirector := proxy.Director
	proxy.Director = func(req *http.Request) {
		originalDirector(req)
		req.Host = target.Host
		req.Header.Set(visionFallbackHeader, fallback)
	}
	proxy.ErrorHandler = func(w http.ResponseWriter, r *http.Request, err error) {
		http.Error(w, "vision fallback proxy error: "+err.Error(), http.StatusBadGateway)
	}

	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		return "", nil, err
	}

	srv := &http.Server{Handler: proxy}
	go func() {
		_ = srv.Serve(ln)
	}()

	stop := func() {
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		_ = srv.Shutdown(ctx)
	}

	return "http://" + ln.Addr().String(), stop, nil
}
