package server

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/gin-gonic/gin"
)

func TestHasAnthropicImageContent(t *testing.T) {
	cases := []struct {
		name string
		body string
		want bool
	}{
		{name: "text only", body: `{"model":"m","messages":[{"role":"user","content":"hi"}]}`, want: false},
		{name: "string content blocks", body: `{"model":"m","messages":[{"role":"user","content":[{"type":"text","text":"hi"}]}]}`, want: false},
		{name: "image block", body: `{"model":"m","messages":[{"role":"user","content":[{"type":"text","text":"what?"},{"type":"image","source":{"type":"base64","media_type":"image/png","data":"AAAA"}}]}]}`, want: true},
		{name: "image in tool_result", body: `{"model":"m","messages":[{"role":"user","content":[{"type":"tool_result","tool_use_id":"x","content":[{"type":"image","source":{"type":"base64","media_type":"image/png","data":"AAAA"}}]}]}]}`, want: true},
		{name: "image url source", body: `{"model":"m","messages":[{"role":"user","content":[{"type":"image","source":{"type":"url","url":"https://example.com/a.png"}}]}]}`, want: true},
		{name: "empty body", body: ``, want: false},
		{name: "invalid json", body: `{not json`, want: false},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := hasAnthropicImageContent([]byte(tc.body)); got != tc.want {
				t.Fatalf("hasAnthropicImageContent(%q) = %v, want %v", tc.body, got, tc.want)
			}
		})
	}
}

func TestIsImageNotSupportedError(t *testing.T) {
	cases := []struct {
		name string
		body string
		want bool
	}{
		{name: "exact phrasing", body: `{"error":"this model does not support image input"}`, want: true},
		{name: "does not support images", body: `{"error":"model does not support images"}`, want: true},
		{name: "not support image", body: `{"error":"this model does not support image"}`, want: true},
		{name: "image input not supported", body: `{"error":"image input not supported"}`, want: true},
		{name: "image input is not supported", body: `{"error":"image input is not supported for this model"}`, want: true},
		{name: "unrelated error", body: `{"error":"context length exceeded"}`, want: false},
		{name: "empty", body: ``, want: false},
		{name: "non-json body", body: `this model does not support image input`, want: true},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := isImageNotSupportedError([]byte(tc.body)); got != tc.want {
				t.Fatalf("isImageNotSupportedError(%q) = %v, want %v", tc.body, got, tc.want)
			}
		})
	}
}

// imageRequestBody is a minimal Anthropic /v1/messages request carrying an
// image content block, addressed to the given cloud model.
func imageRequestBody(model string) string {
	return `{"model":"` + model + `","max_tokens":1024,"stream":false,"messages":[{"role":"user","content":[{"type":"text","text":"what is this?"},{"type":"image","source":{"type":"base64","media_type":"image/png","data":"AAAA"}}]}]}`
}

// resetVisionCaches clears the package-level caption and non-vision caches so
// tests are isolated from each other.
func resetVisionCaches() {
	imageCaptionCache.clear()
	nonVisionModels.clear()
}

// newImageFallbackUpstream returns a mock cloud server. respond is invoked with
// the requested model and the number of prior calls to that model (0-based),
// and returns the status code and body to write back. captured records every
// requested model in order.
func newImageFallbackUpstream(t *testing.T, respond func(model string, call int) (int, string)) (*httptest.Server, *[]string) {
	t.Helper()
	var captured []string
	counts := map[string]int{}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		payload, _ := io.ReadAll(r.Body)
		var req struct {
			Model string `json:"model"`
		}
		_ = json.Unmarshal(payload, &req)
		captured = append(captured, req.Model)
		n := counts[req.Model]
		counts[req.Model] = n + 1
		status, body := respond(req.Model, n)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(status)
		_, _ = w.Write([]byte(body))
	}))
	t.Cleanup(srv.Close)
	return srv, &captured
}

// TestCloudVisionFallback_CaptionsThenPrimary verifies that a non-vision
// primary model: rejects the image (400), the fallback captions it, and the
// primary answers using the caption. Upstream calls: glm-5.2, minimax-m3, glm-5.2.
func TestCloudVisionFallback_CaptionsThenPrimary(t *testing.T) {
	gin.SetMode(gin.TestMode)
	setTestHome(t, t.TempDir())
	resetVisionCaches()

	t.Setenv("OLLAMA_CLOUD_VISION_FALLBACK", "minimax-m3:cloud")

	upstream, captured := newImageFallbackUpstream(t, func(model string, call int) (int, string) {
		switch model {
		case "glm-5.2":
			if call == 0 {
				return http.StatusBadRequest, `{"error":"this model does not support image input"}`
			}
			return http.StatusOK, `{"message":{"role":"assistant","content":"it is a cat"},"done":true}`
		case "minimax-m3":
			return http.StatusOK, `{"message":{"role":"assistant","content":"a cat on a mat"},"done":true}`
		default:
			return http.StatusBadRequest, `{"error":"unexpected model ` + model + `"}`
		}
	})

	original := cloudProxyBaseURL
	cloudProxyBaseURL = upstream.URL
	t.Cleanup(func() { cloudProxyBaseURL = original })

	s := &Server{}
	router, err := s.GenerateRoutes()
	if err != nil {
		t.Fatal(err)
	}
	local := httptest.NewServer(router)
	defer local.Close()

	resp, err := http.Post(local.URL+"/v1/messages", "application/json", bytes.NewBufferString(imageRequestBody("glm-5.2:cloud")))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)

	if resp.StatusCode != http.StatusOK {
		t.Fatalf("expected status 200 after caption-then-primary, got %d (%s)", resp.StatusCode, string(body))
	}
	if !strings.Contains(string(body), "it is a cat") {
		t.Fatalf("expected primary response content in body, got %q", string(body))
	}
	if got := *captured; len(got) != 3 || got[0] != "glm-5.2" || got[1] != "minimax-m3" || got[2] != "glm-5.2" {
		t.Fatalf("expected upstream calls [glm-5.2, minimax-m3, glm-5.2], got %v", got)
	}
}

// TestCloudVisionFallback_ImageCapableModelWorksDirectly verifies that an
// image-capable primary sees the real pixels and answers directly (one call,
// no captioning).
func TestCloudVisionFallback_ImageCapableModelWorksDirectly(t *testing.T) {
	gin.SetMode(gin.TestMode)
	setTestHome(t, t.TempDir())
	resetVisionCaches()

	t.Setenv("OLLAMA_CLOUD_VISION_FALLBACK", "minimax-m3:cloud")

	upstream, captured := newImageFallbackUpstream(t, func(model string, call int) (int, string) {
		if model != "llama3.2-vision" {
			return http.StatusBadRequest, `{"error":"unexpected model ` + model + `"}`
		}
		return http.StatusOK, `{"message":{"role":"assistant","content":"a cat"},"done":true}`
	})

	original := cloudProxyBaseURL
	cloudProxyBaseURL = upstream.URL
	t.Cleanup(func() { cloudProxyBaseURL = original })

	s := &Server{}
	router, err := s.GenerateRoutes()
	if err != nil {
		t.Fatal(err)
	}
	local := httptest.NewServer(router)
	defer local.Close()

	resp, err := http.Post(local.URL+"/v1/messages", "application/json", bytes.NewBufferString(imageRequestBody("llama3.2-vision:cloud")))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)

	if resp.StatusCode != http.StatusOK {
		t.Fatalf("expected status 200, got %d (%s)", resp.StatusCode, string(body))
	}
	if !strings.Contains(string(body), "a cat") {
		t.Fatalf("expected response content in body, got %q", string(body))
	}
	if got := *captured; len(got) != 1 || got[0] != "llama3.2-vision" {
		t.Fatalf("expected single upstream call to llama3.2-vision, got %v", got)
	}
	if nonVisionModels.isNonVision("llama3.2-vision") {
		t.Fatal("image-capable primary should not be marked non-vision")
	}
}

// TestCloudVisionFallback_NoFallbackSurfacesError verifies that without a
// fallback configured, the primary's image rejection surfaces to the client.
func TestCloudVisionFallback_NoFallbackSurfacesError(t *testing.T) {
	gin.SetMode(gin.TestMode)
	setTestHome(t, t.TempDir())
	resetVisionCaches()

	upstream, captured := newImageFallbackUpstream(t, func(model string, call int) (int, string) {
		return http.StatusBadRequest, `{"error":"this model does not support image input"}`
	})

	original := cloudProxyBaseURL
	cloudProxyBaseURL = upstream.URL
	t.Cleanup(func() { cloudProxyBaseURL = original })

	s := &Server{}
	router, err := s.GenerateRoutes()
	if err != nil {
		t.Fatal(err)
	}
	local := httptest.NewServer(router)
	defer local.Close()

	resp, err := http.Post(local.URL+"/v1/messages", "application/json", bytes.NewBufferString(imageRequestBody("glm-5.2:cloud")))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)

	if resp.StatusCode != http.StatusBadRequest {
		t.Fatalf("expected status 400, got %d (%s)", resp.StatusCode, string(body))
	}
	if !strings.Contains(string(body), "does not support image input") {
		t.Fatalf("expected image error in body, got %q", string(body))
	}
	if got := *captured; len(got) != 1 {
		t.Fatalf("expected single upstream call (no fallback), got %v", got)
	}
}

// TestCloudVisionFallback_CachesAcrossTurns verifies that after the first
// image turn populates the non-vision and caption caches, a second identical
// image turn makes only a single primary call (no rejection, no captioning).
func TestCloudVisionFallback_CachesAcrossTurns(t *testing.T) {
	gin.SetMode(gin.TestMode)
	setTestHome(t, t.TempDir())
	resetVisionCaches()

	t.Setenv("OLLAMA_CLOUD_VISION_FALLBACK", "minimax-m3:cloud")

	upstream, captured := newImageFallbackUpstream(t, func(model string, call int) (int, string) {
		switch model {
		case "glm-5.2":
			if call == 0 {
				return http.StatusBadRequest, `{"error":"this model does not support image input"}`
			}
			return http.StatusOK, `{"message":{"role":"assistant","content":"it is a cat"},"done":true}`
		case "minimax-m3":
			return http.StatusOK, `{"message":{"role":"assistant","content":"a cat on a mat"},"done":true}`
		default:
			return http.StatusBadRequest, `{"error":"unexpected model ` + model + `"}`
		}
	})

	original := cloudProxyBaseURL
	cloudProxyBaseURL = upstream.URL
	t.Cleanup(func() { cloudProxyBaseURL = original })

	s := &Server{}
	router, err := s.GenerateRoutes()
	if err != nil {
		t.Fatal(err)
	}
	local := httptest.NewServer(router)
	defer local.Close()

	doRequest := func() *http.Response {
		resp, err := http.Post(local.URL+"/v1/messages", "application/json", bytes.NewBufferString(imageRequestBody("glm-5.2:cloud")))
		if err != nil {
			t.Fatal(err)
		}
		return resp
	}

	// Turn 1: primary rejects, fallback captions, primary answers.
	resp1 := doRequest()
	body1, _ := io.ReadAll(resp1.Body)
	resp1.Body.Close()
	if resp1.StatusCode != http.StatusOK || !strings.Contains(string(body1), "it is a cat") {
		t.Fatalf("turn 1: expected 200 with content, got %d (%s)", resp1.StatusCode, string(body1))
	}
	if got := *captured; len(got) != 3 {
		t.Fatalf("turn 1: expected 3 upstream calls, got %v", got)
	}

	// Turn 2: caches hit — only a single primary answer call.
	resp2 := doRequest()
	body2, _ := io.ReadAll(resp2.Body)
	resp2.Body.Close()
	if resp2.StatusCode != http.StatusOK || !strings.Contains(string(body2), "it is a cat") {
		t.Fatalf("turn 2: expected 200 with content, got %d (%s)", resp2.StatusCode, string(body2))
	}
	if got := *captured; len(got) != 4 || got[3] != "glm-5.2" {
		t.Fatalf("turn 2: expected exactly one more glm-5.2 call (cached), got %v", got)
	}
}

func TestCloudPassthroughMiddleware_DivertsImageRequests(t *testing.T) {
	gin.SetMode(gin.TestMode)

	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{"ok":true}`))
	}))
	defer upstream.Close()

	original := cloudProxyBaseURL
	cloudProxyBaseURL = upstream.URL
	t.Cleanup(func() { cloudProxyBaseURL = original })

	nextCalled := false
	r := gin.New()
	r.POST("/v1/messages", cloudPassthroughMiddleware("test"), func(c *gin.Context) {
		nextCalled = true
		if !c.GetBool(cloudAnthropicImageKey) {
			t.Fatal("expected cloudAnthropicImageKey to be set for image request")
		}
		c.Status(http.StatusOK)
		_, _ = c.Writer.Write([]byte(`{"ok":true}`))
	})

	t.Run("image request is diverted", func(t *testing.T) {
		nextCalled = false
		req := httptest.NewRequest(http.MethodPost, "/v1/messages", bytes.NewBufferString(imageRequestBody("glm-5.2:cloud")))
		rec := httptest.NewRecorder()
		r.ServeHTTP(rec, req)
		if !nextCalled {
			t.Fatal("expected image request to be diverted to the next handler")
		}
	})

	t.Run("text request is proxied", func(t *testing.T) {
		nextCalled = false
		body := `{"model":"glm-5.2:cloud","messages":[{"role":"user","content":"hi"}]}`
		req := httptest.NewRequest(http.MethodPost, "/v1/messages", bytes.NewBufferString(body))
		rec := httptest.NewRecorder()
		r.ServeHTTP(rec, req)
		if nextCalled {
			t.Fatal("expected text-only request to be raw-proxied, not diverted")
		}
	})
}
