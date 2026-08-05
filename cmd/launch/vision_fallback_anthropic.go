package launch

import (
	"encoding/json"
	"fmt"
	"strings"

	"github.com/ollama/ollama/anthropic"
)

// parseMessagesRequest decodes an Anthropic Messages API request body. The
// anthropic.MessageParam.UnmarshalJSON normalizes a plain string content into
// a single text content block, so callers can treat content uniformly as
// []ContentBlock.
func parseMessagesRequest(body []byte) (anthropic.MessagesRequest, error) {
	var req anthropic.MessagesRequest
	if err := json.Unmarshal(body, &req); err != nil {
		return anthropic.MessagesRequest{}, err
	}
	return req, nil
}

// hasImageContent reports whether any message in req carries an image block,
// including images nested inside tool_result content.
func hasImageContent(req anthropic.MessagesRequest) bool {
	for _, m := range req.Messages {
		if messageHasImage(m) {
			return true
		}
	}
	return false
}

func messageHasImage(m anthropic.MessageParam) bool {
	for _, b := range m.Content {
		if blockHasImage(b) {
			return true
		}
	}
	return false
}

// latestUserMessageHasImage reports whether the most recent user turn carries an
// image. The Messages API is stateless and the client re-sends history each
// turn, so a new image is the one in the last user message — the direct-mode
// handoff rule keys off this to decide whether the vision fallback should
// answer the turn itself or the primary should (with historical images
// captioned). Images only in earlier turns are historical context, not a new
// image this turn.
func latestUserMessageHasImage(req anthropic.MessagesRequest) bool {
	for i := len(req.Messages) - 1; i >= 0; i-- {
		if req.Messages[i].Role == "user" {
			return messageHasImage(req.Messages[i])
		}
	}
	return false
}

// blockHasImage reports whether b is, or contains, an image block.
func blockHasImage(b anthropic.ContentBlock) bool {
	if b.Type == "image" {
		return true
	}
	if b.Type == "tool_result" {
		if blocks, ok := toolResultContentBlocks(b.Content); ok {
			for _, cb := range blocks {
				if cb.Type == "image" {
					return true
				}
			}
		}
	}
	return false
}

// toolResultContentBlocks extracts the []ContentBlock form of a tool_result
// block's content via a JSON round-trip. A tool_result's content is either a
// string (no images) or an array of content blocks; the second return is false
// for the string form or anything that does not decode into content blocks, so
// callers can leave it untouched.
func toolResultContentBlocks(c any) ([]anthropic.ContentBlock, bool) {
	if c == nil {
		return nil, false
	}
	raw, err := json.Marshal(c)
	if err != nil {
		return nil, false
	}
	var blocks []anthropic.ContentBlock
	if err := json.Unmarshal(raw, &blocks); err != nil {
		return nil, false
	}
	return blocks, true
}

// replaceImagesInMessage replaces every image block in msg (top-level and
// nested inside tool_result content) with a text block carrying caption,
// preserving all non-image blocks and the message's structure. All images in
// the message share the single caption, which is the fallback's response to
// seeing the whole message in context.
func replaceImagesInMessage(msg *anthropic.MessageParam, caption string) {
	for i := range msg.Content {
		msg.Content[i] = replaceImagesInBlock(msg.Content[i], caption)
	}
}

func replaceImagesInBlock(b anthropic.ContentBlock, caption string) anthropic.ContentBlock {
	if b.Type == "image" {
		text := caption
		return anthropic.ContentBlock{Type: "text", Text: &text}
	}
	if b.Type == "tool_result" {
		if blocks, ok := toolResultContentBlocks(b.Content); ok {
			for i := range blocks {
				blocks[i] = replaceImagesInBlock(blocks[i], caption)
			}
			b.Content = blocks
		}
	}
	return b
}

// captionSystemPrompt is the directive sent to the fallback in place of the
// primary's system prompt when captioning. The primary's system prompt is
// dropped because it is agent scaffolding (persona, tools, instructions) that
// pushes the captioner to act on or comment on the user's task instead of
// describing the image. The conversation messages are still forwarded, so the
// captioner has the task context; this directive fixes its job to "describe
// faithfully and neutrally" so the primary gets a literal caption rather than
// a meta-comment, an attempted answer, or an upbeat characterization it then
// has to second-guess. Used in caption mode for every image turn, and in
// direct mode for historical images on text-only continuation turns.
const captionSystemPrompt = `You are a vision assistant. The user's latest message contains an image that the primary model cannot see. Your only job is to describe that image so the primary model can act on it.

- Transcribe all visible text verbatim, exactly as it appears (including headings, labels, buttons, and code).
- Note counts, positions, colors, sizes, and layout precisely.
- Describe fine visual detail — rendering, glyphs, spacing, alignment, state — do not gloss over it.
- Use a neutral, clinical tone. Report only what is literally visible. Do not characterize quality, attractiveness, correctness, usefulness, or intent; do not praise, apologize, or soften.
- If text or detail is unreadable or unclear, say "illegible" or "unclear" rather than guessing or filling it in.
- Do not attempt the user's task, do not give instructions, and do not comment on the image itself — only describe its contents.
- Keep the description focused and complete.`

// captionText wraps a fallback response as the text that replaces an image
// block, so the primary sees what the vision model made of the image in
// context rather than the raw pixels.
func captionText(fallback, caption string) string {
	var b strings.Builder
	b.WriteString("[The user attached an image. A vision model (")
	b.WriteString(fallback)
	b.WriteString(") described it:\n")
	b.WriteString(caption)
	b.WriteString("\n]")
	return b.String()
}

// extractAssistantText returns the concatenated text of a non-streaming
// MessagesResponse, i.e. the fallback's caption. Non-text blocks (tool_use,
// thinking) are ignored — a caption is prose.
func extractAssistantText(resp anthropic.MessagesResponse) string {
	var b strings.Builder
	for _, block := range resp.Content {
		if block.Type == "text" && block.Text != nil {
			b.WriteString(*block.Text)
		}
	}
	return strings.TrimSpace(b.String())
}

// buildCaptionRequest builds the /v1/messages request sent to the fallback to
// caption the image(s) in the last message of msgs: the same messages the
// primary would have seen up to and including that message, with the model
// swapped to the fallback, streaming off, response length capped, and
// tools/thinking stripped (the captioner only looks and describes, it does not
// act). The primary's system prompt is replaced with captionSystemPrompt so the
// captioner describes the image faithfully instead of acting as the primary.
// msgs is the (possibly trimmed) context window the caller has already
// prepared.
//
// The forwarded messages are also stripped of tool scaffolding (tool_use,
// tool_result, thinking) via stripToolScaffoldingForCaption: the primary's
// tool-call history is irrelevant to describing an image, and some vision
// models — notably minimax — recognize the tool-call pattern in the history and
// respond by emitting their own native tool-call tokens as text (e.g.
// "<]minimax[>...") instead of a caption, which would then be captured as a
// garbage description. Images nested inside tool_result blocks (Claude Code
// image Reads) are lifted out to plain image blocks so the captioner still sees
// the pixels.
func buildCaptionRequest(req anthropic.MessagesRequest, msgs []anthropic.MessageParam, fallback string) anthropic.MessagesRequest {
	return anthropic.MessagesRequest{
		Model:     fallback,
		Messages:  stripToolScaffoldingForCaption(msgs),
		System:    captionSystemPrompt,
		Stream:    false,
		MaxTokens: captionMaxTokens(req.MaxTokens),
	}
}

// toolCallOmitted replaces assistant tool_use blocks in a caption request, so
// the captioner sees that a tool was called without the tool-call payload that
// primes some models to emit their own tool-call tokens.
const toolCallOmitted = "[tool call omitted]"

// toolResultOmitted replaces a tool_result block (after lifting out any image it
// carried) in a caption request, so the user turn stays non-empty without
// forwarding file-read text that is captioning noise and bloats the context.
const toolResultOmitted = "[tool result omitted]"

// stripToolScaffoldingForCaption returns a copy of msgs with all tool_use,
// tool_result, and thinking blocks removed, so the captioner is sent a clean
// text+image conversation. tool_result blocks that carry an image are replaced
// by the image block itself (lifted out of the tool_result wrapper) so the
// captioner still sees the pixels; every other tool_use/tool_result block
// becomes a short text placeholder so messages stay non-empty and
// user/assistant alternation (which the Messages API requires) is preserved.
// thinking blocks are dropped as captioning noise. The original msgs is not
// mutated.
func stripToolScaffoldingForCaption(msgs []anthropic.MessageParam) []anthropic.MessageParam {
	out := make([]anthropic.MessageParam, len(msgs))
	for i := range msgs {
		out[i].Role = msgs[i].Role
		out[i].Content = stripBlocksForCaption(msgs[i].Content)
		if len(out[i].Content) == 0 {
			t := toolCallOmitted
			out[i].Content = []anthropic.ContentBlock{{Type: "text", Text: &t}}
		}
	}
	return out
}

func stripBlocksForCaption(blocks []anthropic.ContentBlock) []anthropic.ContentBlock {
	var out []anthropic.ContentBlock
	for _, b := range blocks {
		switch b.Type {
		case "tool_use", "server_tool_use":
			t := toolCallOmitted
			out = append(out, anthropic.ContentBlock{Type: "text", Text: &t})
		case "tool_result":
			out = append(out, liftToolResultImages(b)...)
		case "thinking":
			// Dropped: the captioner does not need the primary's reasoning.
		default:
			out = append(out, b)
		}
	}
	return out
}

// liftToolResultImages returns the image blocks carried inside a tool_result
// block (lifted out of the tool_result wrapper), or a single placeholder text
// block when the tool_result carried no image. Non-image content (e.g. file
// read text) is omitted — it is captioning noise and bloats the caption
// context.
func liftToolResultImages(b anthropic.ContentBlock) []anthropic.ContentBlock {
	inner, ok := toolResultContentBlocks(b.Content)
	if !ok {
		t := toolResultOmitted
		return []anthropic.ContentBlock{{Type: "text", Text: &t}}
	}
	var imgs []anthropic.ContentBlock
	for _, cb := range inner {
		if cb.Type == "image" {
			imgs = append(imgs, cb)
		}
	}
	if len(imgs) == 0 {
		t := toolResultOmitted
		return []anthropic.ContentBlock{{Type: "text", Text: &t}}
	}
	return imgs
}

// captionMaxTokens caps the fallback's response length when captioning so a
// caption cannot grow to the primary's (potentially very large) max_tokens,
// while still honoring a smaller primary cap.
func captionMaxTokens(primaryMax int) int {
	if primaryMax > 0 && primaryMax < maxCaptionTokens {
		return primaryMax
	}
	return maxCaptionTokens
}

// encodeMessagesRequest marshals req, failing loudly so callers never forward
// a malformed body.
func encodeMessagesRequest(req anthropic.MessagesRequest) ([]byte, error) {
	body, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("encode messages request: %w", err)
	}
	return body, nil
}
