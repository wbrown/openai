package openai

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"strings"
	"time"

	"github.com/wbrown/llmapi"
)

// StreamRaw sends the streaming request this conversation would send for
// (text, sampling) and writes every raw SSE line to w as it arrives, each
// prefixed with the elapsed time since the request was sent and the gap since
// the previous line. It performs NO SSE parsing and updates no conversation
// state beyond appending the user turn — it surfaces exactly what the server
// puts on the wire, and when.
//
// It exists to diagnose stream timing and shape, which the parsed path hides:
//
//   - time-to-first-token: the elapsed stamp on the first data line. If it
//     exceeds a consumer's idle-token timeout, the consumer cancels a stream
//     the server is still (correctly) prefilling or reasoning on.
//   - mid-stream gaps: the "+gap" between consecutive lines, to spot a real
//     stall versus steady trickle.
//   - reasoning field name: whether a given server streams chain-of-thought in
//     delta.reasoning_content (what parseSSEStream reads), delta.content, or
//     some other field — a mismatch means reasoning bytes arrive but no delta
//     is emitted, so an idle watchdog sees silence and cancels.
//   - message↔token ratio: the end-of-stream TALLY counts reasoning vs content
//     messages and divides the server's completion_tokens by total messages, to
//     show whether the server emits one token per SSE message or batches them.
//
// The request body comes from the same buildRequest path production uses, so
// what you see on the wire is what production sends, including the
// chat_template_kwargs that carry reasoning effort. The body is read with no
// client timeout (postStreaming uses Timeout: 0) and the conversation's
// context, so SetContext cancellation still applies.
func (c *Conversation) StreamRaw(text string, sampling llmapi.Sampling, w io.Writer) error {
	if text != "" {
		c.AddMessage(llmapi.RoleUser, text)
	} else if len(c.Messages) == 0 {
		return fmt.Errorf("cannot stream: no messages in conversation")
	}

	req, err := c.buildRequest(sampling, true)
	if err != nil {
		return err
	}

	if jsonData, mErr := json.MarshalIndent(req, "", "  "); mErr == nil {
		fmt.Fprintf(w, "=== REQUEST → %s ===\n%s\n=== STREAM (elapsed | +gap-since-prev | raw line) ===\n", c.endpoint(), jsonData)
	}

	start := time.Now()
	body, err := c.postStreaming(req)
	if err != nil {
		return fmt.Errorf("post streaming after %s: %w", time.Since(start).Round(time.Millisecond), err)
	}
	defer body.Close()

	scanner := bufio.NewScanner(body)
	scanner.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	var last time.Time
	var reasoningMsgs, reasoningRunes, contentMsgs, contentRunes int
	var sawUsage bool
	var promptTokens, completionTokens int
	for scanner.Scan() {
		now := time.Now()
		line := scanner.Text()
		if line == "" {
			continue // SSE event separator
		}
		gap := ""
		if !last.IsZero() {
			gap = fmt.Sprintf(" +%-8s", now.Sub(last).Round(time.Millisecond))
		}
		last = now
		fmt.Fprintf(w, "[%10s%s] %s\n", time.Since(start).Round(time.Millisecond), gap, line)

		// Tally per-kind message counts and capture the server's usage. Each SSE
		// data event is one "message"; counting reasoning vs content messages and
		// dividing the server's completion_tokens by the total shows whether the
		// server emits one token per message or batches several per message.
		data, ok := strings.CutPrefix(line, "data: ")
		if !ok || data == "[DONE]" {
			continue
		}
		var chunk streamChunk
		if json.Unmarshal([]byte(data), &chunk) != nil {
			continue
		}
		if chunk.Usage != nil {
			sawUsage = true
			promptTokens = chunk.Usage.PromptTokens
			completionTokens = chunk.Usage.CompletionTokens
		}
		for _, ch := range chunk.Choices {
			if r := ch.Delta.ReasoningContent; r != "" {
				reasoningMsgs++
				reasoningRunes += len([]rune(r))
			} else if r := ch.Delta.Reasoning; r != "" {
				reasoningMsgs++
				reasoningRunes += len([]rune(r))
			}
			if ch.Delta.Content != "" {
				contentMsgs++
				contentRunes += len([]rune(ch.Delta.Content))
			}
		}
	}
	if scanErr := scanner.Err(); scanErr != nil {
		return fmt.Errorf("error reading stream after %s: %w", time.Since(start).Round(time.Millisecond), scanErr)
	}
	fmt.Fprintf(w, "=== stream ended after %s ===\n", time.Since(start).Round(time.Millisecond))

	totalMsgs := reasoningMsgs + contentMsgs
	fmt.Fprintf(w, "=== TALLY ===\n")
	fmt.Fprintf(w, "reasoning: %d messages, %d runes\n", reasoningMsgs, reasoningRunes)
	fmt.Fprintf(w, "content:   %d messages, %d runes\n", contentMsgs, contentRunes)
	if sawUsage {
		fmt.Fprintf(w, "server usage: prompt_tokens=%d completion_tokens=%d (completion = reasoning+content, no split)\n", promptTokens, completionTokens)
		if totalMsgs > 0 {
			fmt.Fprintf(w, "%d completion tokens / %d streamed messages = %.2f tokens per message\n", completionTokens, totalMsgs, float64(completionTokens)/float64(totalMsgs))
		}
	}
	return nil
}
