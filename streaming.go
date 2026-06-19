package openai

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/wbrown/llmapi"
)

// SendStreaming sends a message with real-time token streaming via SSE. The
// callback is invoked with each text fragment as it arrives, and once with
// ("", true) when the stream completes.
func (c *Conversation) SendStreaming(text string, sampling llmapi.Sampling, callback llmapi.StreamCallback) (
	reply, stopReason string,
	inputTokens, outputTokens, cacheCreationTokens, cacheReadTokens int,
	err error,
) {
	if text != "" {
		c.AddMessage(llmapi.RoleUser, text)
	} else if len(c.Messages) == 0 {
		return "", "", 0, 0, 0, 0, fmt.Errorf("cannot generate: no messages in conversation")
	}

	req, err := c.buildRequest(sampling, true)
	if err != nil {
		return "", "", 0, 0, 0, 0, err
	}
	body, err := c.postStreaming(req)
	if err != nil {
		return "", "", 0, 0, 0, 0, err
	}
	defer body.Close()

	replyText, toolCalls, rawStop, in, out, cached, err := parseSSEStream(body, callback)
	if err != nil {
		return replyText, normalizeFinishReason(rawStop), in, out, 0, cached, err
	}

	c.Messages = append(c.Messages, chatMessage{
		Role:      "assistant",
		Content:   assistantContent(replyText),
		ToolCalls: toolCalls,
	})
	c.Usage.InputTokens += in
	c.Usage.OutputTokens += out
	c.Usage.CacheReadTokens += cached

	return replyText, normalizeFinishReason(rawStop), in, out, 0, cached, nil
}

// SendStreamingUntilDone combines streaming with automatic continuation. It
// streams tokens via callback throughout and continues with a "Continue." user
// message until stopReason != "max_tokens".
func (c *Conversation) SendStreamingUntilDone(text string, sampling llmapi.Sampling, callback llmapi.StreamCallback) (
	reply, stopReason string,
	inputTokens, outputTokens, cacheCreationTokens, cacheReadTokens int,
	err error,
) {
	var total strings.Builder
	input := text
	for {
		var part string
		var inTok, outTok, ccTok, crTok int
		part, stopReason, inTok, outTok, ccTok, crTok, err = c.SendStreaming(input, sampling, callback)
		if err != nil {
			return total.String(), stopReason, inputTokens, outputTokens, cacheCreationTokens, cacheReadTokens, err
		}
		total.WriteString(part)
		inputTokens += inTok
		outputTokens += outTok
		cacheCreationTokens += ccTok
		cacheReadTokens += crTok

		c.MergeIfLastTwoAssistant()

		if stopReason != "max_tokens" {
			break
		}
		input = "Continue."
	}
	return total.String(), stopReason, inputTokens, outputTokens, cacheCreationTokens, cacheReadTokens, nil
}

// SendRichStreaming sends rich content with streaming and returns the full
// response, reconstructing content blocks (text plus any tool calls) from the
// assistant message appended by SendStreaming.
func (c *Conversation) SendRichStreaming(content []llmapi.ContentBlock, sampling llmapi.Sampling, callback llmapi.StreamCallback) (*llmapi.RichResponse, error) {
	if len(content) > 0 {
		c.AddRichMessage(llmapi.RoleUser, content)
	}
	_, stopReason, in, out, _, cached, err := c.SendStreaming("", sampling, callback)
	if err != nil {
		return nil, err
	}

	last := c.Messages[len(c.Messages)-1]
	blocks := contentToBlocks(last.Content)
	for _, tc := range last.ToolCalls {
		blocks = append(blocks, toolCallToBlock(tc))
	}
	return &llmapi.RichResponse{
		Content:                  blocks,
		StopReason:               stopReason,
		InputTokens:              in,
		OutputTokens:             out,
		CacheCreationInputTokens: 0,
		CacheReadInputTokens:     cached,
	}, nil
}

// postStreaming sends a streaming request and returns the response body for SSE
// parsing. It uses a client with no timeout so the stream is not cut short.
func (c *Conversation) postStreaming(req chatCompletionRequest) (io.ReadCloser, error) {
	jsonData, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("error marshaling request: %w", err)
	}

	client := &http.Client{Timeout: 0}
	if c.HttpClient != nil && c.HttpClient.Transport != nil {
		client.Transport = c.HttpClient.Transport
	}

	var resp *http.Response
	var lastErr error
	for attempt := 0; attempt <= retries; attempt++ {
		httpReq, reqErr := http.NewRequestWithContext(c.context(), "POST", c.endpoint(), bytes.NewReader(jsonData))
		if reqErr != nil {
			return nil, fmt.Errorf("error creating request: %w", reqErr)
		}
		httpReq.Header.Set("Content-Type", "application/json")
		if c.ApiToken != "" {
			httpReq.Header.Set("Authorization", "Bearer "+c.ApiToken)
		}
		httpReq.Header.Set("Accept", "text/event-stream")

		resp, lastErr = client.Do(httpReq)
		if lastErr == nil {
			break
		}
		if attempt < retries {
			time.Sleep(retryDelay)
		}
	}
	if lastErr != nil {
		return nil, fmt.Errorf("HTTP error after %d retries: %w", retries, lastErr)
	}
	if resp == nil {
		return nil, fmt.Errorf("HTTP response is nil")
	}
	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		resp.Body.Close()
		return nil, fmt.Errorf("API error (status %d): %s", resp.StatusCode, body)
	}
	return resp.Body, nil
}

// parseSSEStream reads OpenAI chat.completion.chunk events, accumulating text
// deltas (forwarded to the callback), assembling streamed tool calls by index,
// and capturing the finish reason and the trailing usage chunk.
func parseSSEStream(body io.Reader, callback llmapi.StreamCallback) (
	text string,
	toolCalls []toolCall,
	stopReason string,
	inputTokens, outputTokens, cacheReadTokens int,
	err error,
) {
	scanner := bufio.NewScanner(body)
	scanner.Buffer(make([]byte, 0, 64*1024), 1024*1024)

	var textBuilder strings.Builder
	toolByIndex := map[int]*toolCall{}
	var toolOrder []int

	for scanner.Scan() {
		line := scanner.Text()
		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		data := strings.TrimPrefix(line, "data: ")
		if data == "[DONE]" {
			if callback != nil {
				callback(llmapi.StreamDelta{Done: true})
			}
			break
		}

		var chunk streamChunk
		if jsonErr := json.Unmarshal([]byte(data), &chunk); jsonErr != nil {
			continue // tolerate keep-alive/comment lines
		}

		if chunk.Usage != nil {
			inputTokens = chunk.Usage.PromptTokens
			outputTokens = chunk.Usage.CompletionTokens
			cacheReadTokens = chunk.Usage.PromptTokensDetails.CachedTokens
		}
		for _, choice := range chunk.Choices {
			// Reasoning models (e.g. vLLM-served GLM/DeepSeek launched with a
			// reasoning parser) stream their chain-of-thought in a separate field —
			// reasoning_content (DeepSeek/older vLLM) or reasoning (vLLM GLM 0.23+,
			// OpenRouter). Surface whichever is present through the callback tagged
			// as TokenReasoning so consumers can route it (and so an idle-token
			// watchdog sees the stream is alive during a long reasoning phase), but
			// do NOT write it to textBuilder: the returned reply is generated
			// content only — reasoning is not part of it.
			reasoning := choice.Delta.ReasoningContent
			if reasoning == "" {
				reasoning = choice.Delta.Reasoning
			}
			if reasoning != "" && callback != nil {
				callback(llmapi.StreamDelta{Text: reasoning, Kind: llmapi.TokenReasoning})
			}
			if choice.Delta.Content != "" {
				textBuilder.WriteString(choice.Delta.Content)
				if callback != nil {
					callback(llmapi.StreamDelta{Text: choice.Delta.Content, Kind: llmapi.TokenContent})
				}
			}
			for _, tc := range choice.Delta.ToolCalls {
				acc, ok := toolByIndex[tc.Index]
				if !ok {
					acc = &toolCall{Index: tc.Index, Type: "function"}
					toolByIndex[tc.Index] = acc
					toolOrder = append(toolOrder, tc.Index)
				}
				if tc.ID != "" {
					acc.ID = tc.ID
				}
				if tc.Type != "" {
					acc.Type = tc.Type
				}
				if tc.Function.Name != "" {
					acc.Function.Name = tc.Function.Name
				}
				acc.Function.Arguments += tc.Function.Arguments
			}
			if choice.FinishReason != nil && *choice.FinishReason != "" {
				stopReason = *choice.FinishReason
			}
		}
	}
	if scanErr := scanner.Err(); scanErr != nil {
		return textBuilder.String(), nil, stopReason, inputTokens, outputTokens, cacheReadTokens, fmt.Errorf("error reading stream: %w", scanErr)
	}

	for _, idx := range toolOrder {
		toolCalls = append(toolCalls, *toolByIndex[idx])
	}
	return textBuilder.String(), toolCalls, stopReason, inputTokens, outputTokens, cacheReadTokens, nil
}
