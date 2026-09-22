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
// ("", true) when the stream completes. On a mid-stream read error the
// returned values carry what arrived before it, beside the error.
func (c *Conversation) SendStreaming(text string, sampling llmapi.Sampling, callback llmapi.StreamCallback) (
	reply, stopReason string,
	inputTokens, outputTokens, cacheCreationTokens, cacheReadTokens int,
	err error,
) {
	acct, err := c.streamExchange(text, sampling, callback)
	reply, stopReason, inputTokens, outputTokens, cacheCreationTokens, cacheReadTokens = acct.tuple()
	return reply, stopReason, inputTokens, outputTokens, cacheCreationTokens, cacheReadTokens, err
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
// response: the content blocks (text plus any tool calls) and the account of
// the request that produced them.
func (c *Conversation) SendRichStreaming(content []llmapi.ContentBlock, sampling llmapi.Sampling, callback llmapi.StreamCallback) (*llmapi.RichResponse, error) {
	if len(content) > 0 {
		c.AddRichMessage(llmapi.RoleUser, content)
	}
	acct, err := c.streamExchange("", sampling, callback)
	if err != nil {
		return nil, err
	}
	return acct.richResponse(), nil
}

// streamExchange is the streaming twin of exchange: it adds the user text
// (when non-empty), sends the request as an SSE stream, appends the assistant
// reply to history, accumulates usage, and returns the request's account. On
// a mid-stream read error the account carries what arrived before it, beside
// the error, and history and usage are left untouched.
func (c *Conversation) streamExchange(text string, sampling llmapi.Sampling, callback llmapi.StreamCallback) (requestAccount, error) {
	if text != "" {
		c.AddMessage(llmapi.RoleUser, text)
	} else if len(c.Messages) == 0 {
		return requestAccount{}, fmt.Errorf("cannot generate: no messages in conversation")
	}

	req, err := c.buildRequest(sampling, true)
	if err != nil {
		return requestAccount{}, err
	}
	body, err := c.postStreaming(req)
	if err != nil {
		return requestAccount{}, err
	}
	defer body.Close()

	acct, err := parseSSEStream(body, callback)
	acct.budget = req.MaxCompletionTokens
	if err != nil {
		return acct, err
	}

	c.Messages = append(c.Messages, chatMessage{
		Role:      "assistant",
		Content:   assistantContent(acct.text),
		ToolCalls: acct.toolCalls,
	})
	c.accumulateUsage(acct.usage)
	return acct, nil
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

// parseSSEStream reads OpenAI chat.completion.chunk events and returns the
// server-reported half of the request's account: the content deltas
// accumulated into the reply (and forwarded to the callback), the streamed
// tool calls assembled by index, the finish reason verbatim, the trailing
// usage chunk, and the output tokens attributed per channel. The caller fills
// in the budget it sent.
//
// A chunk's token_ids belong to the channel of the text the chunk carries. A
// chunk carrying ids and no text (a channel's closing token, the
// end-of-sequence token) belongs to the channel open when it arrived, content
// before any text has arrived. When no chunk carried ids, the usage chunk's
// reasoning_tokens attributes the split; when the stream carried neither, the
// split is unknown.
func parseSSEStream(body io.Reader, callback llmapi.StreamCallback) (requestAccount, error) {
	scanner := bufio.NewScanner(body)
	scanner.Buffer(make([]byte, 0, 64*1024), 1024*1024)

	var acct requestAccount
	var textBuilder strings.Builder
	toolByIndex := map[int]*toolCall{}
	var toolOrder []int

	var reasoningIDs, contentIDs int
	sawIDs := false
	channel := llmapi.TokenContent

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
			acct.usage = *chunk.Usage
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
			if reasoning != "" {
				channel = llmapi.TokenReasoning
				if callback != nil {
					callback(llmapi.StreamDelta{Text: reasoning, Kind: llmapi.TokenReasoning})
				}
			}
			if choice.Delta.Content != "" {
				channel = llmapi.TokenContent
				textBuilder.WriteString(choice.Delta.Content)
				if callback != nil {
					callback(llmapi.StreamDelta{Text: choice.Delta.Content, Kind: llmapi.TokenContent})
				}
			}
			if len(choice.TokenIDs) > 0 {
				sawIDs = true
				if channel == llmapi.TokenReasoning {
					reasoningIDs += len(choice.TokenIDs)
				} else {
					contentIDs += len(choice.TokenIDs)
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
				acct.finishReason = *choice.FinishReason
			}
		}
	}
	acct.text = textBuilder.String()
	if scanErr := scanner.Err(); scanErr != nil {
		return acct, fmt.Errorf("error reading stream: %w", scanErr)
	}

	for _, idx := range toolOrder {
		acct.toolCalls = append(acct.toolCalls, *toolByIndex[idx])
	}
	if sawIDs {
		acct.split = llmapi.OutputTokenSplit{Reasoning: reasoningIDs, Content: contentIDs, Known: true}
	} else {
		acct.split = splitFromUsage(acct.usage)
	}
	return acct, nil
}
