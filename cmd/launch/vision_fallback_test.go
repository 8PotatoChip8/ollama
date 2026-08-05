package launch

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"

	"github.com/ollama/ollama/anthropic"
	modelpkg "github.com/ollama/ollama/types/model"
)

// fakeMessagesCall records one POST /v1/messages the fake server received.
type fakeMessagesCall struct {
	model    string
	hasImage bool
	system   string // normalized to text for assertion
}

// fakeOllamaServer is a minimal stand-in for the Ollama server: it answers
// /api/show with per-model capabilities and context length, and /v1/messages
// with a 400 image-not-supported for non-vision models sent an image, 200
// otherwise. It records every /v1/messages call so tests can assert routing.
type fakeOllamaServer struct {
	mu                   sync.Mutex
	vision               map[string]bool
	contextLength        map[string]int
	failFirstFallbackImg int // fail this many leading image-to-vision-model calls
	fallbackImageCalls   int
	messagesCalls        []fakeMessagesCall
	showCalls            int
}

func newFakeOllamaServer() *fakeOllamaServer {
	return &fakeOllamaServer{
		vision:        make(map[string]bool),
		contextLength: make(map[string]int),
	}
}

func (f *fakeOllamaServer) handler(t *testing.T) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		switch r.URL.Path {
		case "/api/show":
			var showReq struct {
				Model string `json:"model"`
				Name  string `json:"name"`
			}
			_ = json.Unmarshal(body, &showReq)
			model := showReq.Model
			if model == "" {
				model = showReq.Name
			}
			f.mu.Lock()
			f.showCalls++
			caps := []modelpkg.Capability{modelpkg.CapabilityTools, modelpkg.CapabilityThinking}
			if f.vision[model] {
				caps = append(caps, modelpkg.CapabilityVision)
			}
			ctxLen := f.contextLength[model]
			f.mu.Unlock()
			resp := map[string]any{
				"capabilities": caps,
				"model_info":   map[string]any{"llama.context_length": float64(ctxLen)},
			}
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode(resp)
		case "/v1/messages":
			var req anthropic.MessagesRequest
			_ = json.Unmarshal(body, &req)
			hasImg := hasImageContent(req)
			f.mu.Lock()
			f.messagesCalls = append(f.messagesCalls, fakeMessagesCall{model: req.Model, hasImage: hasImg, system: systemText(req.System)})
			vision := f.vision[req.Model]
			f.mu.Unlock()
			if hasImg && !vision {
				w.WriteHeader(http.StatusBadRequest)
				_, _ = w.Write([]byte(`{"error":"this model does not support image input"}`))
				return
			}
			if hasImg && vision {
				f.mu.Lock()
				f.fallbackImageCalls++
				n := f.fallbackImageCalls
				failFirst := f.failFirstFallbackImg
				f.mu.Unlock()
				if n <= failFirst {
					w.WriteHeader(http.StatusInternalServerError)
					_, _ = w.Write([]byte(`{"error":"caption model unavailable"}`))
					return
				}
			}
			text := "ok from " + req.Model
			resp := anthropic.MessagesResponse{
				ID:      "msg_test",
				Type:    "message",
				Role:    "assistant",
				Model:   req.Model,
				Content: []anthropic.ContentBlock{{Type: "text", Text: &text}},
			}
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode(resp)
		default:
			t.Errorf("unexpected request to fake server: %s %s", r.Method, r.URL.Path)
			w.WriteHeader(http.StatusNotFound)
		}
	})
}

func (f *fakeOllamaServer) calls() []fakeMessagesCall {
	f.mu.Lock()
	defer f.mu.Unlock()
	out := make([]fakeMessagesCall, len(f.messagesCalls))
	copy(out, f.messagesCalls)
	return out
}

func (f *fakeOllamaServer) showCallCount() int {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.showCalls
}

// startFakeProxy wires a fake ollama server and a vision-fallback proxy pointed
// at it. It returns the proxy URL and a stop function.
func startFakeProxy(t *testing.T, f *fakeOllamaServer, primary, fallback string) (string, func()) {
	t.Helper()
	resetVisionFallbackCaches()
	server := httptest.NewServer(f.handler(t))
	t.Setenv("OLLAMA_HOST", server.URL)
	proxyURL, stop, err := startVisionFallbackProxy(primary, fallback)
	if err != nil {
		server.Close()
		t.Fatalf("startVisionFallbackProxy: %v", err)
	}
	return proxyURL, func() {
		stop()
		server.Close()
	}
}

func postMessages(t *testing.T, proxyURL string, req anthropic.MessagesRequest) *http.Response {
	t.Helper()
	body, err := json.Marshal(req)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	resp, err := http.Post(proxyURL+"/v1/messages", "application/json", bytes.NewReader(body))
	if err != nil {
		t.Fatalf("post: %v", err)
	}
	return resp
}

func imageBlock(data string) anthropic.ContentBlock {
	return anthropic.ContentBlock{Type: "image", Source: &anthropic.ImageSource{Type: "base64", MediaType: "image/png", Data: data}}
}

func textBlock(s string) anthropic.ContentBlock {
	return anthropic.ContentBlock{Type: "text", Text: &s}
}

func userMessage(blocks ...anthropic.ContentBlock) anthropic.MessageParam {
	return anthropic.MessageParam{Role: "user", Content: blocks}
}

// TestVisionFallback_NonVisionPrimaryCaptionsThenPrimary: a non-vision primary
// plus an image triggers a caption call to the fallback (with the image) and a
// follow-up primary call with the image replaced by text.
func TestVisionFallback_NonVisionPrimaryCaptionsThenPrimary(t *testing.T) {
	f := newFakeOllamaServer()
	f.vision["minimax-m3:cloud"] = true
	f.contextLength["minimax-m3:cloud"] = 8192
	const primary, fallback = "glm-5.2:cloud", "minimax-m3:cloud"

	proxyURL, stop := startFakeProxy(t, f, primary, fallback)
	defer stop()

	resp := postMessages(t, proxyURL, anthropic.MessagesRequest{
		Model:     primary,
		MaxTokens: 1024,
		Stream:    false,
		Messages:  []anthropic.MessageParam{userMessage(textBlock("what is this?"), imageBlock("aGVsbG8="))},
	})
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d, want 200", resp.StatusCode)
	}

	calls := f.calls()
	// Expect: one caption call to the fallback (with image), one primary call
	// (text only, no image).
	var fallbackImg, primaryText bool
	for _, c := range calls {
		if c.model == fallback && c.hasImage {
			fallbackImg = true
		}
		if c.model == primary && !c.hasImage {
			primaryText = true
		}
	}
	if !fallbackImg {
		t.Errorf("expected a caption call to fallback %q with image; calls=%+v", fallback, calls)
	}
	if !primaryText {
		t.Errorf("expected a primary call to %q with no image; calls=%+v", primary, calls)
	}
	for _, c := range calls {
		if c.model == primary && c.hasImage {
			t.Errorf("primary was sent the image; it should only have received caption text; calls=%+v", calls)
		}
	}
}

// TestVisionFallback_VisionPrimaryPassesThrough: a vision-capable primary gets
// the image directly; the fallback is never called.
func TestVisionFallback_VisionPrimaryPassesThrough(t *testing.T) {
	f := newFakeOllamaServer()
	f.vision["minimax-m3:cloud"] = true
	f.contextLength["minimax-m3:cloud"] = 8192
	const primary, fallback = "minimax-m3:cloud", "gpt-oss:cloud"

	proxyURL, stop := startFakeProxy(t, f, primary, fallback)
	defer stop()

	resp := postMessages(t, proxyURL, anthropic.MessagesRequest{
		Model:     primary,
		MaxTokens: 1024,
		Stream:    false,
		Messages:  []anthropic.MessageParam{userMessage(textBlock("describe"), imageBlock("aGVsbG8="))},
	})
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d, want 200", resp.StatusCode)
	}

	calls := f.calls()
	if len(calls) != 1 {
		t.Fatalf("expected a single pass-through call, got %+v", calls)
	}
	if calls[0].model != primary || !calls[0].hasImage {
		t.Errorf("expected one primary call with image, got %+v", calls)
	}
}

// TestVisionFallback_CaptionCacheHit: a second turn re-sending the same image
// context hits the caption cache, so the fallback is captioned once across both
// turns (two primary calls, one caption call).
func TestVisionFallback_CaptionCacheHit(t *testing.T) {
	f := newFakeOllamaServer()
	f.vision["minimax-m3:cloud"] = true
	f.contextLength["minimax-m3:cloud"] = 8192
	const primary, fallback = "glm-5.2:cloud", "minimax-m3:cloud"

	proxyURL, stop := startFakeProxy(t, f, primary, fallback)
	defer stop()

	img := imageBlock("aGVsbG8=")
	first := anthropic.MessagesRequest{
		Model: primary, MaxTokens: 1024, Stream: false,
		Messages: []anthropic.MessageParam{userMessage(textBlock("turn 1"), img)},
	}
	resp := postMessages(t, proxyURL, first)
	resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("turn 1 status = %d", resp.StatusCode)
	}

	// Second turn re-sends the same history (including the same image) plus a
	// new trailing exchange, exactly as a stateless client does.
	second := anthropic.MessagesRequest{
		Model: primary, MaxTokens: 1024, Stream: false,
		Messages: []anthropic.MessageParam{
			userMessage(textBlock("turn 1"), img),
			{Role: "assistant", Content: []anthropic.ContentBlock{textBlock("ok from " + primary)}},
			userMessage(textBlock("turn 2")),
		},
	}
	resp2 := postMessages(t, proxyURL, second)
	resp2.Body.Close()
	if resp2.StatusCode != http.StatusOK {
		t.Fatalf("turn 2 status = %d", resp2.StatusCode)
	}

	calls := f.calls()
	var captionCalls, primaryCalls int
	for _, c := range calls {
		if c.model == fallback {
			captionCalls++
		}
		if c.model == primary {
			primaryCalls++
		}
	}
	if captionCalls != 1 {
		t.Errorf("expected exactly one caption call (cache hit on turn 2), got %d; calls=%+v", captionCalls, calls)
	}
	if primaryCalls != 2 {
		t.Errorf("expected two primary calls (one per turn), got %d; calls=%+v", primaryCalls, calls)
	}
}

// TestVisionFallback_ContextOverflowTrims: when the fallback's context window
// is smaller than the caption context, the proxy trims oldest non-system turns
// and still returns 200 rather than looping or failing.
func TestVisionFallback_ContextOverflowTrims(t *testing.T) {
	f := newFakeOllamaServer()
	f.vision["minimax-m3:cloud"] = true
	f.contextLength["minimax-m3:cloud"] = 200 // tiny window
	const primary, fallback = "glm-5.2:cloud", "minimax-m3:cloud"

	proxyURL, stop := startFakeProxy(t, f, primary, fallback)
	defer stop()

	img := imageBlock("aGVsbG8=")
	var msgs []anthropic.MessageParam
	// Many large prior turns so the estimate exceeds the fallback's 200-token
	// window; the image-bearing message is last and must be preserved.
	for i := 0; i < 20; i++ {
		msgs = append(msgs, userMessage(textBlock(bigString(2000))))
	}
	msgs = append(msgs, userMessage(textBlock("finally, this"), img))

	resp := postMessages(t, proxyURL, anthropic.MessagesRequest{
		Model: primary, MaxTokens: 1024, Stream: false, Messages: msgs,
	})
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d, want 200 after trim", resp.StatusCode)
	}
	if len(f.calls()) == 0 {
		t.Fatal("expected at least one /v1/messages call")
	}
}

// TestVisionFallback_DegradeOnCaptionFailure: if the fallback cannot caption,
// the proxy degrades to letting the fallback answer the turn directly with the
// original image.
func TestVisionFallback_DegradeOnCaptionFailure(t *testing.T) {
	f := newFakeOllamaServer()
	f.vision["minimax-m3:cloud"] = true
	f.contextLength["minimax-m3:cloud"] = 8192
	f.failFirstFallbackImg = 1 // the caption call (1st image-to-fallback) fails
	const primary, fallback = "glm-5.2:cloud", "minimax-m3:cloud"

	proxyURL, stop := startFakeProxy(t, f, primary, fallback)
	defer stop()

	resp := postMessages(t, proxyURL, anthropic.MessagesRequest{
		Model: primary, MaxTokens: 1024, Stream: false,
		Messages: []anthropic.MessageParam{userMessage(textBlock("what?"), imageBlock("aGVsbG8="))},
	})
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d, want 200 (degraded to fallback)", resp.StatusCode)
	}
	body, _ := io.ReadAll(resp.Body)
	// The degrade forward sends the original image request to the fallback,
	// which serves it (the 2nd image-to-fallback call, past failFirst=1).
	if !strings.Contains(string(body), fallback) {
		t.Errorf("expected degraded response from fallback %q, got %s", fallback, string(body))
	}
}

// TestVisionFallback_NoImagePassesThrough: a request without an image passes
// straight to the primary; the fallback is never called and /api/show is not
// consulted.
func TestVisionFallback_NoImagePassesThrough(t *testing.T) {
	f := newFakeOllamaServer()
	f.vision["minimax-m3:cloud"] = true
	const primary, fallback = "glm-5.2:cloud", "minimax-m3:cloud"

	proxyURL, stop := startFakeProxy(t, f, primary, fallback)
	defer stop()

	resp := postMessages(t, proxyURL, anthropic.MessagesRequest{
		Model: primary, MaxTokens: 1024, Stream: false,
		Messages: []anthropic.MessageParam{userMessage(textBlock("plain text"))},
	})
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d, want 200", resp.StatusCode)
	}
	calls := f.calls()
	if len(calls) != 1 || calls[0].model != primary || calls[0].hasImage {
		t.Errorf("expected one text-only primary call, got %+v", calls)
	}
	if f.showCallCount() != 0 {
		t.Errorf("expected no /api/show call for an image-less request, got %d", f.showCallCount())
	}
}

// TestVisionFallback_ProxyErrorHandler: a client disconnect (context.Canceled)
// is debug-logged and does NOT produce a 502; a real transport error still does.
func TestVisionFallback_ProxyErrorHandler(t *testing.T) {
	req := httptest.NewRequest(http.MethodPost, "/v1/messages", nil)

	rec := httptest.NewRecorder()
	proxyErrorHandler(rec, req, context.Canceled)
	if rec.Code != http.StatusOK {
		t.Errorf("context.Canceled should not write an error response, got status %d body %q", rec.Code, rec.Body.String())
	}

	rec2 := httptest.NewRecorder()
	proxyErrorHandler(rec2, req, errors.New("boom"))
	if rec2.Code != http.StatusBadGateway {
		t.Errorf("real error should be 502, got status %d body %q", rec2.Code, rec2.Body.String())
	}

	// A wrapped context.Canceled (as net/http hands the proxy on client
	// disconnect) is also treated as a cancellation, not an error.
	rec3 := httptest.NewRecorder()
	proxyErrorHandler(rec3, req, fmt.Errorf("Get %q: %w", "http://127.0.0.1:1/v1/messages", context.Canceled))
	if rec3.Code != http.StatusOK {
		t.Errorf("wrapped context.Canceled should not write an error response, got status %d", rec3.Code)
	}
}

func bigString(n int) string {
	b := make([]byte, n)
	for i := range b {
		b[i] = 'x'
	}
	return string(b)
}

// systemText normalizes a MessagesRequest.System value (string, or an array of
// content blocks as decoded from JSON) to plain text for assertions.
func systemText(s any) string {
	switch v := s.(type) {
	case string:
		return v
	case []any:
		var b strings.Builder
		for _, item := range v {
			if m, ok := item.(map[string]any); ok {
				if t, ok := m["text"].(string); ok {
					b.WriteString(t)
				}
			}
		}
		return b.String()
	}
	return ""
}

// TestVisionFallback_CaptionUsesDescribeDirective: the caption call to the
// fallback must carry the describe-image system directive, NOT the primary's
// system prompt (which is agent scaffolding that would push the captioner to
// act or comment instead of describing). The primary call keeps its own system.
func TestVisionFallback_CaptionUsesDescribeDirective(t *testing.T) {
	f := newFakeOllamaServer()
	f.vision["minimax-m3:cloud"] = true
	f.contextLength["minimax-m3:cloud"] = 8192
	const primary, fallback = "glm-5.2:cloud", "minimax-m3:cloud"
	const primarySystem = "You are Claude Code, an interactive coding agent. Use tools."

	proxyURL, stop := startFakeProxy(t, f, primary, fallback)
	defer stop()

	resp := postMessages(t, proxyURL, anthropic.MessagesRequest{
		Model:     primary,
		MaxTokens: 1024,
		Stream:    false,
		System:    primarySystem,
		Messages:  []anthropic.MessageParam{userMessage(textBlock("what is this?"), imageBlock("aGVsbG8="))},
	})
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d, want 200", resp.StatusCode)
	}

	var captionSys, primarySys string
	for _, c := range f.calls() {
		if c.model == fallback && c.hasImage {
			captionSys = c.system
		}
		if c.model == primary && !c.hasImage {
			primarySys = c.system
		}
	}
	if captionSys == "" {
		t.Fatal("expected a caption call to the fallback with an image")
	}
	if captionSys != captionSystemPrompt {
		t.Errorf("caption call system should be the describe directive, got %q", captionSys)
	}
	if captionSys == primarySystem {
		t.Errorf("caption call system leaked the primary's system prompt: %q", captionSys)
	}
	if primarySys != primarySystem {
		t.Errorf("primary call system should keep the primary's system %q, got %q", primarySystem, primarySys)
	}
}
