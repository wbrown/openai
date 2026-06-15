package openai

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/wbrown/llmapi"
)

// Compile-time interface check.
var _ llmapi.Conversation = (*Conversation)(nil)

// DefaultBaseURL is the OpenAI API base URL (the "/v1" root); the request path
// "/chat/completions" is appended to it. Override per conversation via
// SetEndpoint to target Azure OpenAI or any OpenAI-compatible server (vLLM,
// Together, Groq, OpenRouter, llama.cpp, ...) — pass the base URL ending in /v1.
var DefaultBaseURL = "https://api.openai.com/v1"

// DefaultApiToken is loaded from OPENAI_API_KEY (or a token file) in init().
// It seeds the ApiToken of every conversation created with NewConversation and
// can be overridden directly or per conversation.
var DefaultApiToken string

// HTTP retry configuration.
var (
	retries    = 3
	retryDelay = 3 * time.Second
)

// Conversation manages a chat session against an OpenAI Chat Completions
// endpoint.
type Conversation struct {
	// Ctx is the context for cancellation and timeouts. If nil,
	// context.Background() is used.
	Ctx context.Context
	// System is the system prompt, prepended as a system-role message on every
	// request. It is not stored in Messages.
	System string
	// Messages is the conversation history in OpenAI wire form.
	Messages []chatMessage
	// Usage tracks cumulative token consumption.
	Usage Usage
	// ApiToken is the bearer token for this conversation.
	ApiToken string
	// Settings configures generation parameters.
	Settings Settings
	// HttpClient is used for API requests.
	HttpClient *http.Client
	// Tools are the tool definitions offered to the model.
	Tools []llmapi.ToolDefinition
	// Endpoint overrides DefaultBaseURL when non-empty. It is a base URL (the
	// "/v1" root); "/chat/completions" is appended to it per request.
	Endpoint string
}

// NewConversation creates a conversation seeded with DefaultSettings and
// DefaultApiToken. The model is intentionally unset (see Settings.Model).
func NewConversation(system string) *Conversation {
	return &Conversation{
		System:     system,
		Messages:   make([]chatMessage, 0),
		ApiToken:   DefaultApiToken,
		Settings:   DefaultSettings,
		HttpClient: &http.Client{Timeout: 120 * time.Second},
	}
}

// context returns the conversation's context, defaulting to Background if nil.
func (c *Conversation) context() context.Context {
	if c.Ctx != nil {
		return c.Ctx
	}
	return context.Background()
}

// endpoint returns the full chat-completions URL: the configured base URL (or
// DefaultBaseURL) with "/chat/completions" appended.
func (c *Conversation) endpoint() string {
	base := c.Endpoint
	if base == "" {
		base = DefaultBaseURL
	}
	return strings.TrimRight(base, "/") + "/chat/completions"
}

// SetEndpoint overrides the API base URL (the "/v1" root, e.g.
// "http://host:8000/v1"); "/chat/completions" is appended per request. Pass ""
// to revert to DefaultBaseURL.
func (c *Conversation) SetEndpoint(endpoint string) {
	c.Endpoint = endpoint
}

// ==========================================================================
// Request construction
// ==========================================================================

// requestMessages returns the full messages array sent to the API, with the
// system prompt prepended as a system-role message.
func (c *Conversation) requestMessages() []chatMessage {
	msgs := make([]chatMessage, 0, len(c.Messages)+1)
	if c.System != "" {
		msgs = append(msgs, chatMessage{Role: "system", Content: c.System})
	}
	return append(msgs, c.Messages...)
}

// buildRequest assembles the request body. It returns an error when no model is
// configured — model selection is the caller's decision, not the library's.
func (c *Conversation) buildRequest(sampling llmapi.Sampling, stream bool) (chatCompletionRequest, error) {
	if c.Settings.Model == "" {
		return chatCompletionRequest{}, fmt.Errorf("model not set: set Settings.Model or call SetModel")
	}

	temp, topP := resolveSampling(c.Settings, sampling)
	req := chatCompletionRequest{
		Model:               c.Settings.Model,
		Messages:            c.requestMessages(),
		MaxCompletionTokens: c.Settings.MaxTokens,
		Temperature:         temp,
		TopP:                topP,
		Stop:                c.Settings.StopSequences,
	}
	if c.Settings.FrequencyPenalty != 0 {
		v := c.Settings.FrequencyPenalty
		req.FrequencyPenalty = &v
	}
	if c.Settings.PresencePenalty != 0 {
		v := c.Settings.PresencePenalty
		req.PresencePenalty = &v
	}
	if c.Settings.Seed != 0 {
		v := c.Settings.Seed
		req.Seed = &v
	}
	if len(c.Tools) > 0 {
		req.Tools = toOpenAITools(c.Tools)
	}
	if stream {
		req.Stream = true
		req.StreamOptions = &streamOptions{IncludeUsage: true}
	}
	return req, nil
}

// resolveSampling layers per-call overrides over conversation defaults and
// returns pointers that are nil when the value is zero, so JSON marshalling
// omits them and the server applies its own default. A non-zero override
// replaces the configured value; the llmapi.Sampling.TopK field has no standard
// chat-completions equivalent and is ignored.
func resolveSampling(s Settings, o llmapi.Sampling) (temperature, topP *float64) {
	t := s.Temperature
	if o.Temperature != 0 {
		t = o.Temperature
	}
	p := s.TopP
	if o.TopP != 0 {
		p = o.TopP
	}
	if t != 0 {
		temperature = &t
	}
	if p != 0 {
		topP = &p
	}
	return temperature, topP
}

// toOpenAITools converts llmapi tool definitions to OpenAI's function-tool form.
func toOpenAITools(defs []llmapi.ToolDefinition) []tool {
	result := make([]tool, len(defs))
	for i, d := range defs {
		result[i] = tool{
			Type: "function",
			Function: toolFunction{
				Name:        d.Name,
				Description: d.Description,
				Parameters:  d.InputSchema,
			},
		}
	}
	return result
}

// ==========================================================================
// HTTP
// ==========================================================================

// postRequest marshals and sends a non-streaming request, retrying transport
// errors, and returns the parsed response.
func (c *Conversation) postRequest(req chatCompletionRequest) (*chatCompletionResponse, error) {
	jsonData, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("error marshaling request: %w", err)
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

		resp, lastErr = c.HttpClient.Do(httpReq)
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
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("error reading response: %w", err)
	}
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("API error (status %d): %s", resp.StatusCode, body)
	}

	var cr chatCompletionResponse
	if err := json.Unmarshal(body, &cr); err != nil {
		return nil, fmt.Errorf("error parsing response: %w", err)
	}
	if cr.Error != nil {
		return nil, fmt.Errorf("API error: %s", cr.Error.Message)
	}
	if len(cr.Choices) == 0 {
		return nil, fmt.Errorf("no choices in response")
	}
	return &cr, nil
}

// exchange adds the user text (when non-empty), sends the conversation, appends
// the assistant reply to history, and accumulates usage. With empty text it
// resends the current history (OpenAI chat has no assistant-prefill), erroring
// only when there is nothing to send.
func (c *Conversation) exchange(text string, sampling llmapi.Sampling) (*chatCompletionResponse, error) {
	if text != "" {
		c.AddMessage(llmapi.RoleUser, text)
	} else if len(c.Messages) == 0 {
		return nil, fmt.Errorf("cannot generate: no messages in conversation")
	}

	req, err := c.buildRequest(sampling, false)
	if err != nil {
		return nil, err
	}
	cr, err := c.postRequest(req)
	if err != nil {
		return nil, err
	}

	msg := cr.Choices[0].Message
	c.Messages = append(c.Messages, chatMessage{
		Role:      "assistant",
		Content:   assistantContent(msg.Content),
		ToolCalls: msg.ToolCalls,
	})
	c.accumulateUsage(cr.Usage)
	return cr, nil
}

// accumulateUsage folds one response's usage into the conversation total.
func (c *Conversation) accumulateUsage(u usage) {
	c.Usage.InputTokens += u.PromptTokens
	c.Usage.OutputTokens += u.CompletionTokens
	c.Usage.CacheReadTokens += u.PromptTokensDetails.CachedTokens
}

// ==========================================================================
// Send
// ==========================================================================

// Send sends a user message and returns the assistant's reply. cacheCreationTokens
// is always 0 (OpenAI does not bill cache writes); cacheReadTokens reflects the
// automatic prompt-cache hit reported in usage.prompt_tokens_details.
func (c *Conversation) Send(text string, sampling llmapi.Sampling) (
	reply, stopReason string,
	inputTokens, outputTokens, cacheCreationTokens, cacheReadTokens int,
	err error,
) {
	cr, err := c.exchange(text, sampling)
	if err != nil {
		return "", "", 0, 0, 0, 0, err
	}
	choice := cr.Choices[0]
	return choice.Message.Content,
		normalizeFinishReason(choice.FinishReason),
		cr.Usage.PromptTokens,
		cr.Usage.CompletionTokens,
		0,
		cr.Usage.PromptTokensDetails.CachedTokens,
		nil
}

// SendUntilDone repeatedly calls Send until stopReason != "max_tokens",
// returning the accumulated output. Continuation is driven by a "Continue."
// user message, since chat completions cannot continue an assistant turn.
func (c *Conversation) SendUntilDone(text string, sampling llmapi.Sampling) (
	reply, stopReason string,
	inputTokens, outputTokens, cacheCreationTokens, cacheReadTokens int,
	err error,
) {
	var total strings.Builder
	input := text
	for {
		var part string
		var inTok, outTok, ccTok, crTok int
		part, stopReason, inTok, outTok, ccTok, crTok, err = c.Send(input, sampling)
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

// ==========================================================================
// Rich content
// ==========================================================================

// SendRich sends rich content blocks and returns the full response. If content
// is empty it continues from the current history.
func (c *Conversation) SendRich(content []llmapi.ContentBlock, sampling llmapi.Sampling) (*llmapi.RichResponse, error) {
	if len(content) > 0 {
		c.AddRichMessage(llmapi.RoleUser, content)
	}
	cr, err := c.exchange("", sampling)
	if err != nil {
		return nil, err
	}
	return responseToRich(cr), nil
}

// responseToRich converts a parsed response into an llmapi.RichResponse.
func responseToRich(cr *chatCompletionResponse) *llmapi.RichResponse {
	choice := cr.Choices[0]
	return &llmapi.RichResponse{
		Content:                  messageToBlocks(choice.Message),
		StopReason:               normalizeFinishReason(choice.FinishReason),
		InputTokens:              cr.Usage.PromptTokens,
		OutputTokens:             cr.Usage.CompletionTokens,
		CacheCreationInputTokens: 0,
		CacheReadInputTokens:     cr.Usage.PromptTokensDetails.CachedTokens,
	}
}

// messageToBlocks converts an assistant response message into content blocks:
// a text block (when present) plus one tool_use block per tool call.
func messageToBlocks(m responseMessage) []llmapi.ContentBlock {
	var blocks []llmapi.ContentBlock
	if m.Content != "" {
		blocks = append(blocks, llmapi.NewTextBlock(m.Content))
	}
	for _, tc := range m.ToolCalls {
		blocks = append(blocks, toolCallToBlock(tc))
	}
	return blocks
}

// toolCallToBlock converts an OpenAI tool call into an llmapi tool_use block.
func toolCallToBlock(tc toolCall) llmapi.ContentBlock {
	return llmapi.ContentBlock{
		Type: llmapi.ContentTypeToolUse,
		ToolUse: &llmapi.ToolUseContent{
			ID:    tc.ID,
			Name:  tc.Function.Name,
			Input: json.RawMessage(tc.Function.Arguments),
		},
	}
}

// AddRichMessage adds a message with multiple content blocks. Text and image
// blocks form one message of the given role; tool_use blocks become its
// tool_calls; each tool_result block becomes its own role:"tool" message.
// Thinking and document blocks are not representable in chat-completions input
// and are dropped (see GetCapabilities).
func (c *Conversation) AddRichMessage(role llmapi.Role, content []llmapi.ContentBlock) {
	var parts []contentPart
	var toolCalls []toolCall
	var plainText strings.Builder
	hasImage := false

	for _, block := range content {
		switch block.Type {
		case llmapi.ContentTypeText:
			plainText.WriteString(block.Text)
			parts = append(parts, contentPart{Type: "text", Text: block.Text})
		case llmapi.ContentTypeImage:
			if block.Image != nil {
				hasImage = true
				parts = append(parts, contentPart{
					Type:     "image_url",
					ImageURL: &imageURL{URL: imageDataURL(block.Image.Source)},
				})
			}
		case llmapi.ContentTypeToolUse:
			if block.ToolUse != nil {
				toolCalls = append(toolCalls, toolCall{
					ID:   block.ToolUse.ID,
					Type: "function",
					Function: functionCall{
						Name:      block.ToolUse.Name,
						Arguments: string(block.ToolUse.Input),
					},
				})
			}
		case llmapi.ContentTypeToolResult:
			if block.ToolResult != nil {
				c.Messages = append(c.Messages, chatMessage{
					Role:       "tool",
					ToolCallID: block.ToolResult.ToolUseID,
					Content:    block.ToolResult.Content,
				})
			}
		}
	}

	if len(parts) == 0 && len(toolCalls) == 0 {
		return
	}
	msg := chatMessage{Role: string(role), ToolCalls: toolCalls}
	switch {
	case hasImage:
		msg.Content = parts
	case len(parts) > 0:
		msg.Content = plainText.String()
	default:
		msg.Content = nil // tool-call-only assistant turn -> "content":null
	}
	c.Messages = append(c.Messages, msg)
}

// GetRichMessages returns the conversation history as rich messages. A
// role:"tool" message maps back to a user message carrying a tool_result block,
// matching how llmapi represents tool results.
func (c *Conversation) GetRichMessages() []llmapi.RichMessage {
	result := make([]llmapi.RichMessage, 0, len(c.Messages))
	for _, m := range c.Messages {
		result = append(result, chatMessageToRich(m))
	}
	return result
}

// chatMessageToRich converts one stored message back to an llmapi.RichMessage.
func chatMessageToRich(m chatMessage) llmapi.RichMessage {
	if m.Role == "tool" {
		return llmapi.RichMessage{
			Role: llmapi.RoleUser,
			Content: []llmapi.ContentBlock{
				llmapi.NewToolResultBlock(m.ToolCallID, contentString(m.Content), false),
			},
		}
	}
	blocks := contentToBlocks(m.Content)
	for _, tc := range m.ToolCalls {
		blocks = append(blocks, toolCallToBlock(tc))
	}
	return llmapi.RichMessage{Role: llmapi.Role(m.Role), Content: blocks}
}

// ==========================================================================
// Content conversion
// ==========================================================================

// assistantContent stores an assistant reply as a string, or nil (serialized as
// "content":null) when empty so a tool-call-only turn is well-formed.
func assistantContent(s string) any {
	if s == "" {
		return nil
	}
	return s
}

// imageDataURL renders an llmapi image source as an OpenAI image_url value.
func imageDataURL(src llmapi.ImageSource) string {
	if src.Type == "url" {
		return src.URL
	}
	return fmt.Sprintf("data:%s;base64,%s", src.MediaType, src.Data)
}

// imageURLToBlock parses an OpenAI image_url back into an llmapi image block,
// decoding the "data:<mediatype>;base64,<data>" form when present.
func imageURLToBlock(url string) llmapi.ContentBlock {
	const marker = ";base64,"
	if rest, ok := strings.CutPrefix(url, "data:"); ok {
		if i := strings.Index(rest, marker); i >= 0 {
			return llmapi.NewImageBlock(llmapi.MediaType(rest[:i]), rest[i+len(marker):])
		}
	}
	return llmapi.NewImageBlockFromURL("", url)
}

// contentToBlocks converts a stored Content value into llmapi content blocks.
func contentToBlocks(content any) []llmapi.ContentBlock {
	switch v := content.(type) {
	case string:
		if v == "" {
			return nil
		}
		return []llmapi.ContentBlock{llmapi.NewTextBlock(v)}
	case []contentPart:
		var blocks []llmapi.ContentBlock
		for _, p := range v {
			switch p.Type {
			case "text":
				blocks = append(blocks, llmapi.NewTextBlock(p.Text))
			case "image_url":
				if p.ImageURL != nil {
					blocks = append(blocks, imageURLToBlock(p.ImageURL.URL))
				}
			}
		}
		return blocks
	}
	return nil
}

// contentToText flattens a stored Content value to plain text.
func contentToText(content any) string {
	switch v := content.(type) {
	case string:
		return v
	case []contentPart:
		var b strings.Builder
		for _, p := range v {
			if p.Type == "text" {
				b.WriteString(p.Text)
			}
		}
		return b.String()
	}
	return ""
}

// contentString returns the string content of a message, or "" if not a string.
func contentString(content any) string {
	if s, ok := content.(string); ok {
		return s
	}
	return ""
}

// normalizeFinishReason maps OpenAI finish reasons to the common vocabulary
// shared with the anthropic and novelai siblings.
func normalizeFinishReason(reason string) string {
	switch reason {
	case "stop":
		return "end_turn"
	case "length":
		return "max_tokens"
	case "tool_calls", "function_call":
		return "tool_use"
	default:
		return reason
	}
}

// ==========================================================================
// History management
// ==========================================================================

// AddMessage adds a plain-text message to the history.
func (c *Conversation) AddMessage(role llmapi.Role, content string) {
	c.Messages = append(c.Messages, chatMessage{Role: string(role), Content: content})
}

// GetMessages returns the history as simple role/text messages.
func (c *Conversation) GetMessages() []llmapi.Message {
	result := make([]llmapi.Message, 0, len(c.Messages))
	for _, m := range c.Messages {
		result = append(result, llmapi.Message{
			Role:    llmapi.Role(m.Role),
			Content: contentToText(m.Content),
		})
	}
	return result
}

// MergeIfLastTwoAssistant merges two trailing plain-text assistant messages into
// one. Messages carrying tool_calls or non-string content are left untouched.
func (c *Conversation) MergeIfLastTwoAssistant() {
	n := len(c.Messages)
	if n < 2 {
		return
	}
	last := c.Messages[n-1]
	prev := c.Messages[n-2]
	if last.Role != "assistant" || prev.Role != "assistant" {
		return
	}
	if len(last.ToolCalls) > 0 || len(prev.ToolCalls) > 0 {
		return
	}
	lastText, ok1 := last.Content.(string)
	prevText, ok2 := prev.Content.(string)
	if !ok1 || !ok2 {
		return
	}
	merged := strings.TrimRight(prevText, " \t\n\r") + strings.TrimSpace(lastText)
	c.Messages[n-2].Content = merged
	c.Messages = c.Messages[:n-1]
}

// ==========================================================================
// Accessors and configuration
// ==========================================================================

// GetUsage returns cumulative token usage. CacheCreationInputTokens is always 0;
// CacheReadInputTokens reflects OpenAI's automatic prompt-cache hits.
func (c *Conversation) GetUsage() llmapi.Usage {
	return llmapi.Usage{
		InputTokens:              c.Usage.InputTokens,
		OutputTokens:             c.Usage.OutputTokens,
		CacheCreationInputTokens: 0,
		CacheReadInputTokens:     c.Usage.CacheReadTokens,
	}
}

// GetSystem returns the system prompt.
func (c *Conversation) GetSystem() string {
	return c.System
}

// Clear resets the history and usage, preserving the system prompt and settings.
func (c *Conversation) Clear() {
	c.Messages = make([]chatMessage, 0)
	c.Usage = Usage{}
}

// SetContext sets the context for cancellation and timeouts. Pass nil to revert
// to context.Background().
func (c *Conversation) SetContext(ctx context.Context) {
	c.Ctx = ctx
}

// SetModel changes the model for subsequent API calls.
func (c *Conversation) SetModel(model string) {
	c.Settings.Model = model
}

// SetTools configures the available tools. Pass nil or empty to disable tools.
func (c *Conversation) SetTools(tools []llmapi.ToolDefinition) {
	c.Tools = tools
}

// GetTools returns the currently configured tools.
func (c *Conversation) GetTools() []llmapi.ToolDefinition {
	return c.Tools
}

// GetCapabilities reports the Chat Completions feature set. Documents and
// thinking are not surfaced by this endpoint.
func (c *Conversation) GetCapabilities() llmapi.Capabilities {
	return llmapi.Capabilities{
		SupportsImages:      true,
		SupportsDocuments:   false,
		SupportsToolUse:     true,
		SupportsThinking:    false,
		SupportsStreaming:   true,
		SupportsCaching:     true,
		MaxImageSize:        20 * 1024 * 1024,
		SupportedImageTypes: []string{"image/png", "image/jpeg", "image/gif", "image/webp"},
	}
}

// EnableSystemCaching is a no-op: OpenAI caches eligible prompts (>=1024 tokens)
// automatically with no client control. Returns nil because caching is supported
// and active; cache hits surface as CacheReadInputTokens.
func (c *Conversation) EnableSystemCaching() error { return nil }

// EnableConversationCaching is a no-op for the same reason as EnableSystemCaching.
func (c *Conversation) EnableConversationCaching() error { return nil }

// DisableConversationCaching is a no-op: OpenAI's automatic caching cannot be
// turned off from the client.
func (c *Conversation) DisableConversationCaching() error { return nil }

// ==========================================================================
// Token loading
// ==========================================================================

// init loads the API token. Priority: OPENAI_API_KEY > ~/.openai_key > ./.openai_key.
func init() {
	if token := os.Getenv("OPENAI_API_KEY"); token != "" {
		DefaultApiToken = token
		return
	}
	if home, err := os.UserHomeDir(); err == nil {
		if token := readTokenFile(home + "/.openai_key"); token != "" {
			DefaultApiToken = token
			return
		}
	}
	if token := readTokenFile(".openai_key"); token != "" {
		DefaultApiToken = token
	}
}

// readTokenFile reads a token from a file, returning "" on error.
func readTokenFile(path string) string {
	data, err := os.ReadFile(path)
	if err != nil {
		return ""
	}
	return strings.TrimSpace(string(data))
}
