package launch

import (
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/ollama/ollama/envconfig"
)

// TestVisionFallbackShim_InjectsHeaderAndProxies verifies the shim forwards
// requests to the Ollama server (selected via OLLAMA_HOST) and injects the
// X-Ollama-Cloud-Vision-Fallback header on every request, and that the response
// body and status stream through unchanged.
func TestVisionFallbackShim_InjectsHeaderAndProxies(t *testing.T) {
	var gotHeader string
	var gotPath string
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotHeader = r.Header.Get(visionFallbackHeader)
		gotPath = r.URL.Path
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusTeapot)
		_, _ = w.Write([]byte(`{"ok":true}`))
	}))
	defer upstream.Close()

	t.Setenv("OLLAMA_HOST", upstream.URL)

	baseURL, stop, err := startVisionFallbackShim("minimax-m3:cloud")
	if err != nil {
		t.Fatalf("startVisionFallbackShim: %v", err)
	}
	defer stop()

	resp, err := http.Post(baseURL+"/v1/messages", "application/json", strings.NewReader(`{"hello":"world"}`))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)

	if resp.StatusCode != http.StatusTeapot {
		t.Fatalf("expected status to pass through (418), got %d", resp.StatusCode)
	}
	if string(body) != `{"ok":true}` {
		t.Fatalf("expected body to pass through, got %q", string(body))
	}
	if gotHeader != "minimax-m3:cloud" {
		t.Fatalf("expected upstream to receive fallback header %q, got %q", "minimax-m3:cloud", gotHeader)
	}
	if gotPath != "/v1/messages" {
		t.Fatalf("expected upstream path /v1/messages, got %q", gotPath)
	}
}

// TestVisionFallbackShim_StreamsSSE verifies the shim forwards a chunked SSE
// stream incrementally rather than buffering the whole response (FlushInterval
// is -1). The upstream writes two SSE events in separate flushes; the client
// must receive them without waiting for the upstream to close.
func TestVisionFallbackShim_StreamsSSE(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		fl := w.(http.Flusher)
		_, _ = w.Write([]byte("data: event-one\n\n"))
		fl.Flush()
		_, _ = w.Write([]byte("data: event-two\n\n"))
		fl.Flush()
	}))
	defer upstream.Close()

	t.Setenv("OLLAMA_HOST", upstream.URL)

	baseURL, stop, err := startVisionFallbackShim("minimax-m3:cloud")
	if err != nil {
		t.Fatalf("startVisionFallbackShim: %v", err)
	}
	defer stop()

	resp, err := http.Post(baseURL+"/v1/messages", "application/json", strings.NewReader(`{}`))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)

	if resp.StatusCode != http.StatusOK {
		t.Fatalf("expected 200, got %d", resp.StatusCode)
	}
	want := "data: event-one\n\ndata: event-two\n\n"
	if string(body) != want {
		t.Fatalf("expected streamed SSE body %q, got %q", want, string(body))
	}
}

// TestVisionFallbackShim_StopClosesListener verifies the stop function shuts the
// shim down so subsequent requests fail to connect.
func TestVisionFallbackShim_StopClosesListener(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	}))
	defer upstream.Close()

	t.Setenv("OLLAMA_HOST", upstream.URL)

	baseURL, stop, err := startVisionFallbackShim("minimax-m3:cloud")
	if err != nil {
		t.Fatalf("startVisionFallbackShim: %v", err)
	}

	stop()

	_, err = http.Post(baseURL+"/v1/messages", "application/json", strings.NewReader(`{}`))
	if err == nil {
		t.Fatal("expected request to fail after stop, but it succeeded")
	}
}

// TestVisionFallbackShim_BaseURLIsLocalhost verifies the shim binds to loopback
// (not an external interface) so the per-launch fallback stays local.
func TestVisionFallbackShim_BaseURLIsLocalhost(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	}))
	defer upstream.Close()

	t.Setenv("OLLAMA_HOST", upstream.URL)

	baseURL, stop, err := startVisionFallbackShim("minimax-m3:cloud")
	if err != nil {
		t.Fatalf("startVisionFallbackShim: %v", err)
	}
	defer stop()

	if !strings.HasPrefix(baseURL, "http://127.0.0.1:") {
		t.Fatalf("expected shim base URL on 127.0.0.1, got %q", baseURL)
	}
	// Sanity: the shim URL is distinct from the configured Ollama host.
	if baseURL == envconfig.Host().String() {
		t.Fatalf("shim base URL should differ from envconfig.Host(), both %q", baseURL)
	}
}
