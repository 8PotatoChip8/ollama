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

// captionText wraps a fallback response as the text that replaces an image
// block, so the primary sees what the vision model made of the image in
// context rather than the raw pixels.
func captionText(fallback, caption string) string {
	var b strings.Builder
	b.WriteString("[The user attached an image. A vision model (")
	b.WriteString(fallback)
	b.WriteString(") was given the full conversation up to this point — exactly as if it were the primary model — and responded:\n")
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
// tools/thinking stripped (the captioner only looks and responds, it does not
// act). msgs is the (possibly trimmed) context window the caller has already
// prepared.
func buildCaptionRequest(req anthropic.MessagesRequest, msgs []anthropic.MessageParam, fallback string) anthropic.MessagesRequest {
	return anthropic.MessagesRequest{
		Model:     fallback,
		Messages:  msgs,
		System:    req.System,
		Stream:    false,
		MaxTokens: captionMaxTokens(req.MaxTokens),
	}
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
