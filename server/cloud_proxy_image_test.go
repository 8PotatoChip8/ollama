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
		{name: "image after string-content turns", body: `{"model":"m","system":"s","messages":[{"role":"user","content":"hi"},{"role":"assistant","content":"ok"},{"role":"user","content":[{"type":"text","text":"what?"},{"type":"image","source":{"type":"base64","media_type":"image/png","data":"AAAA"}}]}]}`, want: true},
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
	return imageRequestBodyWithText(model, "what is this?")
}

// imageRequestBodyWithText is a minimal Anthropic /v1/messages request carrying
// an image content block alongside the given user text, addressed to the given
// cloud model. The text is what the captioner should receive as context.
func imageRequestBodyWithText(model, text string) string {
	return `{"model":"` + model + `","max_tokens":1024,"stream":false,"messages":[{"role":"user","content":[{"type":"text","text":"` + text + `"},{"type":"image","source":{"type":"base64","media_type":"image/png","data":"AAAA"}}]}]}`
}

// resetVisionCaches clears the package-level caption and non-vision caches so
// tests are isolated from each other.
func resetVisionCaches() {
	imageCaptionCache.clear()
	nonVisionModels.clear()
	cloudModelInfoCache.clear()
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
		if r.URL.Path == "/api/show" {
			// No context_length in model_info -> captioner treats the window as
			// unknown and skips proactive trimming (falls back to reactive retry).
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusOK)
			_, _ = w.Write([]byte(`{"model_info":{"test.arch":"x"}}`))
			return
		}
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

// capturedReq records a single upstream request's model and the contents of
// every message it carried, so tests can assert what the captioner was given.
type capturedReq struct {
	Model    string
	Contents []string
}

// newImageFallbackUpstreamCapturing is like newImageFallbackUpstream but also
// records the content of every message in each request, so tests can verify
// the caption request carries the full conversation context (system prompt +
// prior turns + the image-bearing message), not just the image.
func newImageFallbackUpstreamCapturing(t *testing.T, respond func(model string, call int) (int, string)) (*httptest.Server, *[]capturedReq) {
	t.Helper()
	var captured []capturedReq
	counts := map[string]int{}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/api/show" {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusOK)
			_, _ = w.Write([]byte(`{"model_info":{"test.arch":"x"}}`))
			return
		}
		payload, _ := io.ReadAll(r.Body)
		var req struct {
			Model    string `json:"model"`
			Messages []struct {
				Role    string `json:"role"`
				Content string `json:"content"`
			} `json:"messages"`
		}
		_ = json.Unmarshal(payload, &req)
		contents := make([]string, len(req.Messages))
		for i, m := range req.Messages {
			contents[i] = m.Content
		}
		captured = append(captured, capturedReq{Model: req.Model, Contents: contents})
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

// TestCloudVisionFallback_ProactiveCapabilitySkipsPrimaryPixels verifies that
// when /api/show reports the primary has no vision capability, the doomed
// primary-with-pixels attempt is skipped entirely: the fallback captions and
// the primary answers, with no wasted first 400 call. Upstream /api/chat calls:
// minimax-m3, glm-5.2 (two, not three).
func TestCloudVisionFallback_ProactiveCapabilitySkipsPrimaryPixels(t *testing.T) {
	gin.SetMode(gin.TestMode)
	setTestHome(t, t.TempDir())
	resetVisionCaches()

	t.Setenv("OLLAMA_CLOUD_VISION_FALLBACK", "minimax-m3:cloud")

	var captured []string
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/api/show" {
			payload, _ := io.ReadAll(r.Body)
			var sr struct {
				Model string `json:"model"`
			}
			_ = json.Unmarshal(payload, &sr)
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusOK)
			switch sr.Model {
			case "glm-5.2":
				// Primary declares NO vision capability.
				_, _ = w.Write([]byte(`{"model_info":{"glm5.2.context_length":1000000},"capabilities":["completion","tools","thinking"]}`))
			case "minimax-m3":
				_, _ = w.Write([]byte(`{"model_info":{"minimax-m3.context_length":524288},"capabilities":["completion","tools","thinking","vision"]}`))
			default:
				_, _ = w.Write([]byte(`{"model_info":{}}`))
			}
			return
		}
		payload, _ := io.ReadAll(r.Body)
		var req struct {
			Model string `json:"model"`
		}
		_ = json.Unmarshal(payload, &req)
		captured = append(captured, req.Model)
		w.Header().Set("Content-Type", "application/json")
		switch req.Model {
		case "glm-5.2":
			// Primary is only called once here (to answer after captioning),
			// since the proactive capability check skipped the pixels attempt.
			w.WriteHeader(http.StatusOK)
			_, _ = w.Write([]byte(`{"message":{"role":"assistant","content":"answered"},"done":true}`))
		case "minimax-m3":
			w.WriteHeader(http.StatusOK)
			_, _ = w.Write([]byte(`{"message":{"role":"assistant","content":"a red square"},"done":true}`))
		default:
			w.WriteHeader(http.StatusBadRequest)
			_, _ = w.Write([]byte(`{"error":"unexpected model ` + req.Model + `"}`))
		}
	}))
	defer upstream.Close()

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
		t.Fatalf("expected 200, got %d (%s)", resp.StatusCode, string(body))
	}
	if !strings.Contains(string(body), "answered") {
		t.Fatalf("expected primary answer in body, got %q", string(body))
	}
	// No doomed primary-pixels call: only [minimax-m3 (caption), glm-5.2 (answer)].
	if got := captured; len(got) != 2 || got[0] != "minimax-m3" || got[1] != "glm-5.2" {
		t.Fatalf("expected upstream /api/chat calls [minimax-m3, glm-5.2] (proactive skip), got %v", got)
	}
	if !nonVisionModels.isNonVision("glm-5.2") {
		t.Fatal("primary with no vision capability should be marked non-vision proactively")
	}
}

// TestCloudVisionFallback_ProactiveCapabilityAllowsVisionPrimary verifies the
// flip side: when /api/show reports the primary HAS vision capability, the
// proactive check lets it through to see the real pixels and answer directly
// (one call, no captioning, not marked non-vision).
func TestCloudVisionFallback_ProactiveCapabilityAllowsVisionPrimary(t *testing.T) {
	gin.SetMode(gin.TestMode)
	setTestHome(t, t.TempDir())
	resetVisionCaches()

	t.Setenv("OLLAMA_CLOUD_VISION_FALLBACK", "minimax-m3:cloud")

	var captured []string
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/api/show" {
			payload, _ := io.ReadAll(r.Body)
			var sr struct {
				Model string `json:"model"`
			}
			_ = json.Unmarshal(payload, &sr)
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusOK)
			switch sr.Model {
			case "llama3.2-vision":
				_, _ = w.Write([]byte(`{"model_info":{"llama3.2-vision.context_length":131072},"capabilities":["completion","tools","vision"]}`))
			default:
				_, _ = w.Write([]byte(`{"model_info":{}}`))
			}
			return
		}
		payload, _ := io.ReadAll(r.Body)
		var req struct {
			Model string `json:"model"`
		}
		_ = json.Unmarshal(payload, &req)
		captured = append(captured, req.Model)
		w.Header().Set("Content-Type", "application/json")
		if req.Model != "llama3.2-vision" {
			w.WriteHeader(http.StatusBadRequest)
			_, _ = w.Write([]byte(`{"error":"unexpected model ` + req.Model + `"}`))
			return
		}
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{"message":{"role":"assistant","content":"a cat"},"done":true}`))
	}))
	defer upstream.Close()

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
		t.Fatalf("expected 200, got %d (%s)", resp.StatusCode, string(body))
	}
	if !strings.Contains(string(body), "a cat") {
		t.Fatalf("expected direct answer in body, got %q", string(body))
	}
	if got := captured; len(got) != 1 || got[0] != "llama3.2-vision" {
		t.Fatalf("expected a single direct primary call, got %v", got)
	}
	if nonVisionModels.isNonVision("llama3.2-vision") {
		t.Fatal("vision-capable primary should not be marked non-vision")
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

// imageRequestBodyRich is an Anthropic /v1/messages request with a system
// prompt and a prior user/assistant turn before the image-bearing user turn,
// so tests can verify the captioner receives the full conversation context up
// to the image — exactly as if the fallback were the primary model.
func imageRequestBodyRich(model string) string {
	return `{"model":"` + model + `","max_tokens":1024,"stream":false,` +
		`"system":"You are a debugging assistant.",` +
		`"messages":[` +
		`{"role":"user","content":"Let's debug an error together."},` +
		`{"role":"assistant","content":"Sure, share what you're seeing."},` +
		`{"role":"user","content":[` +
		`{"type":"text","text":"what error is shown in this screenshot?"},` +
		`{"type":"image","source":{"type":"base64","media_type":"image/png","data":"AAAA"}}` +
		`]}]}`
}

// TestCloudVisionFallback_CaptionUsesRequestContext verifies that the caption
// request sent to the fallback vision model carries the full conversation up
// to the image — the system prompt, prior turns, and the image-bearing message
// — exactly as if the fallback were the primary model, so its caption reflects
// the full task context rather than a context-free "describe everything" prompt.
func TestCloudVisionFallback_CaptionUsesRequestContext(t *testing.T) {
	gin.SetMode(gin.TestMode)
	setTestHome(t, t.TempDir())
	resetVisionCaches()

	t.Setenv("OLLAMA_CLOUD_VISION_FALLBACK", "minimax-m3:cloud")

	upstream, captured := newImageFallbackUpstreamCapturing(t, func(model string, call int) (int, string) {
		switch model {
		case "glm-5.2":
			if call == 0 {
				return http.StatusBadRequest, `{"error":"this model does not support image input"}`
			}
			return http.StatusOK, `{"message":{"role":"assistant","content":"answered"},"done":true}`
		case "minimax-m3":
			return http.StatusOK, `{"message":{"role":"assistant","content":"a red square"},"done":true}`
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

	resp, err := http.Post(local.URL+"/v1/messages", "application/json",
		bytes.NewBufferString(imageRequestBodyRich("glm-5.2:cloud")))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		t.Fatalf("expected 200, got %d (%s)", resp.StatusCode, string(body))
	}

	// Find the caption request (the one sent to the fallback model) and assert
	// it carries the full conversation context: system, prior user, prior
	// assistant, and the image-bearing user message.
	var captionReq *capturedReq
	for i := range *captured {
		if (*captured)[i].Model == "minimax-m3" {
			captionReq = &(*captured)[i]
			break
		}
	}
	if captionReq == nil {
		t.Fatalf("expected a caption request to minimax-m3, got %v", *captured)
	}
	wantContents := []string{
		"You are a debugging assistant.",
		"Let's debug an error together.",
		"Sure, share what you're seeing.",
		"what error is shown in this screenshot?",
	}
	if len(captionReq.Contents) != len(wantContents) {
		t.Fatalf("caption request should carry %d messages (full context), got %d: %v", len(wantContents), len(captionReq.Contents), captionReq.Contents)
	}
	for i, want := range wantContents {
		if captionReq.Contents[i] != want {
			t.Fatalf("caption request message %d = %q, want %q (full context not passed: %v)", i, captionReq.Contents[i], want, captionReq.Contents)
		}
	}
}

// TestCloudVisionFallback_CaptionTrimsOnContextOverflow verifies that when the
// fallback's context window is smaller than the conversation up to the image,
// the captioner progressively drops the oldest non-system turns — preserving the
// system prompt and the image-bearing message — and retries until the request
// fits, then returns a caption. The fallback here only accepts system + image
// (2 messages); the rich request has 4, so it must trim twice before succeeding.
func TestCloudVisionFallback_CaptionTrimsOnContextOverflow(t *testing.T) {
	gin.SetMode(gin.TestMode)
	setTestHome(t, t.TempDir())
	resetVisionCaches()

	t.Setenv("OLLAMA_CLOUD_VISION_FALLBACK", "minimax-m3:cloud")

	counts := map[string]int{}
	var minimaxMsgCounts []int
	var lastMinimaxContents []string
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/api/show" {
			// No context_length -> proactive trim skipped, so the reactive
			// trim-on-overflow retry is exercised ([4,3,2]).
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusOK)
			_, _ = w.Write([]byte(`{"model_info":{"test.arch":"x"}}`))
			return
		}
		payload, _ := io.ReadAll(r.Body)
		var req struct {
			Model    string `json:"model"`
			Messages []struct {
				Role    string `json:"role"`
				Content string `json:"content"`
			} `json:"messages"`
		}
		_ = json.Unmarshal(payload, &req)
		n := counts[req.Model]
		counts[req.Model] = n + 1
		w.Header().Set("Content-Type", "application/json")
		switch req.Model {
		case "glm-5.2":
			if n == 0 {
				w.WriteHeader(http.StatusBadRequest)
				_, _ = w.Write([]byte(`{"error":"this model does not support image input"}`))
				return
			}
			w.WriteHeader(http.StatusOK)
			_, _ = w.Write([]byte(`{"message":{"role":"assistant","content":"answered"},"done":true}`))
		case "minimax-m3":
			minimaxMsgCounts = append(minimaxMsgCounts, len(req.Messages))
			lastMinimaxContents = nil
			for _, m := range req.Messages {
				lastMinimaxContents = append(lastMinimaxContents, m.Content)
			}
			// Fallback "context window" fits only system + image (2 messages).
			if len(req.Messages) > 2 {
				w.WriteHeader(http.StatusBadRequest)
				_, _ = w.Write([]byte(`{"error":"context length exceeded"}`))
				return
			}
			w.WriteHeader(http.StatusOK)
			_, _ = w.Write([]byte(`{"message":{"role":"assistant","content":"caption"},"done":true}`))
		default:
			w.WriteHeader(http.StatusBadRequest)
			_, _ = w.Write([]byte(`{"error":"unexpected model ` + req.Model + `"}`))
		}
	}))
	defer upstream.Close()

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

	resp, err := http.Post(local.URL+"/v1/messages", "application/json",
		bytes.NewBufferString(imageRequestBodyRich("glm-5.2:cloud")))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("expected 200 after trimming, got %d (%s)", resp.StatusCode, string(body))
	}
	if !strings.Contains(string(body), "answered") {
		t.Fatalf("expected primary answer in body, got %q", string(body))
	}
	// 4 messages -> reject, 3 -> reject, 2 (system + image) -> succeed.
	if got := minimaxMsgCounts; len(got) != 3 || got[0] != 4 || got[1] != 3 || got[2] != 2 {
		t.Fatalf("expected minimax calls with message counts [4,3,2], got %v", got)
	}
	// The final (successful) caption request must still carry the system prompt
	// and the image-bearing message — never the dropped middle turns.
	wantLast := []string{"You are a debugging assistant.", "what error is shown in this screenshot?"}
	if len(lastMinimaxContents) != len(wantLast) {
		t.Fatalf("final caption request should carry system + image message, got %v", lastMinimaxContents)
	}
	for i, want := range wantLast {
		if lastMinimaxContents[i] != want {
			t.Fatalf("final caption request message %d = %q, want %q (system/image not preserved: %v)", i, lastMinimaxContents[i], want, lastMinimaxContents)
		}
	}
}

// TestCloudVisionFallback_CaptionProactivelyTrims verifies that when /api/show
// reports the fallback's context window, the captioner proactively trims the
// conversation to fit that window BEFORE sending — so a too-large context is
// handled in a single caption call (no reactive retries). The mock reports a
// tiny window (50 tokens); the rich 4-message request's estimate exceeds it, so
// it is trimmed down to system + image message and sent once.
func TestCloudVisionFallback_CaptionProactivelyTrims(t *testing.T) {
	gin.SetMode(gin.TestMode)
	setTestHome(t, t.TempDir())
	resetVisionCaches()

	t.Setenv("OLLAMA_CLOUD_VISION_FALLBACK", "minimax-m3:cloud")

	counts := map[string]int{}
	var minimaxMsgCounts []int
	var lastMinimaxContents []string
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/api/show" {
			// Report a tiny context window so proactive trimming kicks in.
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusOK)
			_, _ = w.Write([]byte(`{"model_info":{"test.context_length":50}}`))
			return
		}
		payload, _ := io.ReadAll(r.Body)
		var req struct {
			Model    string `json:"model"`
			Messages []struct {
				Role    string `json:"role"`
				Content string `json:"content"`
			} `json:"messages"`
		}
		_ = json.Unmarshal(payload, &req)
		n := counts[req.Model]
		counts[req.Model] = n + 1
		w.Header().Set("Content-Type", "application/json")
		switch req.Model {
		case "glm-5.2":
			if n == 0 {
				w.WriteHeader(http.StatusBadRequest)
				_, _ = w.Write([]byte(`{"error":"this model does not support image input"}`))
				return
			}
			w.WriteHeader(http.StatusOK)
			_, _ = w.Write([]byte(`{"message":{"role":"assistant","content":"answered"},"done":true}`))
		case "minimax-m3":
			minimaxMsgCounts = append(minimaxMsgCounts, len(req.Messages))
			lastMinimaxContents = nil
			for _, m := range req.Messages {
				lastMinimaxContents = append(lastMinimaxContents, m.Content)
			}
			w.WriteHeader(http.StatusOK)
			_, _ = w.Write([]byte(`{"message":{"role":"assistant","content":"caption"},"done":true}`))
		default:
			w.WriteHeader(http.StatusBadRequest)
			_, _ = w.Write([]byte(`{"error":"unexpected model ` + req.Model + `"}`))
		}
	}))
	defer upstream.Close()

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

	resp, err := http.Post(local.URL+"/v1/messages", "application/json",
		bytes.NewBufferString(imageRequestBodyRich("glm-5.2:cloud")))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("expected 200, got %d (%s)", resp.StatusCode, string(body))
	}
	if !strings.Contains(string(body), "answered") {
		t.Fatalf("expected primary answer in body, got %q", string(body))
	}
	// Proactively trimmed to system + image message in a single caption call —
	// no reactive retries.
	if got := minimaxMsgCounts; len(got) != 1 || got[0] != 2 {
		t.Fatalf("expected a single proactive caption call with 2 messages (system + image), got %v", got)
	}
	wantLast := []string{"You are a debugging assistant.", "what error is shown in this screenshot?"}
	if len(lastMinimaxContents) != len(wantLast) {
		t.Fatalf("proactive caption request should carry system + image message, got %v", lastMinimaxContents)
	}
	for i, want := range wantLast {
		if lastMinimaxContents[i] != want {
			t.Fatalf("proactive caption request message %d = %q, want %q (system/image not preserved: %v)", i, lastMinimaxContents[i], want, lastMinimaxContents)
		}
	}
}

// TestCloudVisionFallback_CacheKeysOnAccompanyingText verifies that the caption
// cache is keyed on (image, accompanying text): the same image sent under a
// different user intent is re-captioned, while the same image+intent repeated
// hits the cache. With the same image throughout, two distinct intents should
// produce exactly two caption calls; repeating the first intent adds none.
func TestCloudVisionFallback_CacheKeysOnAccompanyingText(t *testing.T) {
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
			return http.StatusOK, `{"message":{"role":"assistant","content":"answered"},"done":true}`
		case "minimax-m3":
			return http.StatusOK, `{"message":{"role":"assistant","content":"caption"},"done":true}`
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

	post := func(text string) {
		resp, err := http.Post(local.URL+"/v1/messages", "application/json",
			bytes.NewBufferString(imageRequestBodyWithText("glm-5.2:cloud", text)))
		if err != nil {
			t.Fatal(err)
		}
		defer resp.Body.Close()
		if resp.StatusCode != http.StatusOK {
			body, _ := io.ReadAll(resp.Body)
			t.Fatalf("expected 200 for %q, got %d (%s)", text, resp.StatusCode, string(body))
		}
	}

	// Distinct intents over the same image -> two caption calls.
	post("describe the colors")
	post("read the error text")
	// Repeating the first intent -> cache hit, no new caption call.
	post("describe the colors")

	minimaxCalls := 0
	for _, m := range *captured {
		if m == "minimax-m3" {
			minimaxCalls++
		}
	}
	if minimaxCalls != 2 {
		t.Fatalf("expected exactly 2 caption calls (one per distinct intent), got %d (captured: %v)", minimaxCalls, *captured)
	}
}

// TestCloudVisionFallback_HeaderOverridesEnvVar verifies that a per-launch
// fallback conveyed via the X-Ollama-Cloud-Vision-Fallback header takes
// precedence over the server-wide OLLAMA_CLOUD_VISION_FALLBACK env var, so
// concurrent launches against one server can use different fallbacks. The env
// var names minimax-m3:cloud but the header names qwen-vl:cloud; the captioner
// must be called with qwen-vl, not minimax-m3.
func TestCloudVisionFallback_HeaderOverridesEnvVar(t *testing.T) {
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
		case "qwen-vl":
			return http.StatusOK, `{"message":{"role":"assistant","content":"a cat on a mat"},"done":true}`
		case "minimax-m3":
			// If the env var won, the captioner would be minimax-m3 — fail loudly.
			return http.StatusBadRequest, `{"error":"env var fallback should not have been used"}`
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

	req, err := http.NewRequest(http.MethodPost, local.URL+"/v1/messages", bytes.NewBufferString(imageRequestBody("glm-5.2:cloud")))
	if err != nil {
		t.Fatal(err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("X-Ollama-Cloud-Vision-Fallback", "qwen-vl:cloud")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)

	if resp.StatusCode != http.StatusOK {
		t.Fatalf("expected status 200, got %d (%s)", resp.StatusCode, string(body))
	}
	// Reactive path (generic /api/show reports no capabilities): primary-pixels
	// 400, then captioner, then primary answer. The captioner (index 1) must be
	// the header's model, not the env var's.
	got := *captured
	if len(got) != 3 || got[0] != "glm-5.2" || got[1] != "qwen-vl" || got[2] != "glm-5.2" {
		t.Fatalf("expected upstream calls [glm-5.2, qwen-vl, glm-5.2] (header overrides env var), got %v", got)
	}
}
