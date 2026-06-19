// Package openai provides a client for OpenAI's Chat Completions API
// (/v1/chat/completions) that implements the github.com/wbrown/llmapi
// Conversation interface. It is a sibling of github.com/wbrown/anthropic and
// github.com/wbrown/novelai, allowing code written against llmapi to swap to
// any OpenAI-compatible chat endpoint (OpenAI, Azure OpenAI, vLLM, Together,
// Groq, OpenRouter, llama.cpp servers, ...) by pointing SetEndpoint at it.
package openai

import "encoding/json"

// Settings configures generation parameters for OpenAI Chat Completions.
//
// Fields left at their zero value are omitted from the request body so the
// server applies its own default. This is deliberate: reasoning models
// (o1/o3/gpt-5) reject any non-default Temperature/TopP, so leaving them unset
// is what makes those models work without a per-model capability table.
type Settings struct {
	// Model to use for generation (e.g. "gpt-4o", "gpt-4.1", "o3").
	// There is no default — model selection is the caller's decision. Send
	// returns an error if Model is empty.
	Model string
	// MaxTokens caps generated tokens. Sent as max_completion_tokens (the
	// modern field; max_tokens is deprecated for chat completions and rejected
	// by reasoning models). Omitted when 0.
	MaxTokens int
	// Temperature controls randomness (0.0-2.0). Omitted when 0.
	Temperature float64
	// TopP is nucleus sampling (0.0-1.0). Omitted when 0.
	TopP float64
	// FrequencyPenalty penalizes frequent tokens (-2.0 to 2.0). Omitted when 0.
	FrequencyPenalty float64
	// PresencePenalty penalizes already-present tokens (-2.0 to 2.0). Omitted when 0.
	PresencePenalty float64
	// Seed requests best-effort deterministic sampling. Omitted when 0.
	Seed int
	// StopSequences are strings that stop generation.
	StopSequences []string
}

// DefaultSettings provides reasonable defaults. Model is intentionally empty;
// the caller must set it via Settings.Model or SetModel.
var DefaultSettings = Settings{
	MaxTokens:   2048,
	Temperature: 1.0,
}

// Usage tracks token consumption for a conversation.
type Usage struct {
	InputTokens  int
	OutputTokens int
	// CacheReadTokens is the number of prompt tokens served from OpenAI's
	// automatic prompt cache (usage.prompt_tokens_details.cached_tokens).
	CacheReadTokens int
}

// ==========================================================================
// Request wire structs
// ==========================================================================

// chatCompletionRequest is the /v1/chat/completions request body.
type chatCompletionRequest struct {
	Model               string         `json:"model"`
	Messages            []chatMessage  `json:"messages"`
	MaxCompletionTokens int            `json:"max_completion_tokens,omitempty"`
	Temperature         *float64       `json:"temperature,omitempty"`
	TopP                *float64       `json:"top_p,omitempty"`
	FrequencyPenalty    *float64       `json:"frequency_penalty,omitempty"`
	PresencePenalty     *float64       `json:"presence_penalty,omitempty"`
	Seed                *int           `json:"seed,omitempty"`
	Stop                []string       `json:"stop,omitempty"`
	Tools               []tool         `json:"tools,omitempty"`
	Stream              bool           `json:"stream,omitempty"`
	StreamOptions       *streamOptions `json:"stream_options,omitempty"`
	// ChatTemplateKwargs carries vLLM chat-template kwargs at the request top level
	// (where the OpenAI SDK's extra_body lands). Used to drive reasoning effort:
	// {"enable_thinking": false} to disable, or {"reasoning_effort": "<level>"}.
	ChatTemplateKwargs map[string]any `json:"chat_template_kwargs,omitempty"`
}

// chatMessage is a single message in the request messages array.
//
// Content is polymorphic per the OpenAI spec: a plain string for text-only
// turns, a []contentPart array when images are present, or nil (serialized as
// "content":null) for an assistant turn that carries only tool_calls.
type chatMessage struct {
	Role string `json:"role"` // "system", "user", "assistant", "tool"
	// Content is left without omitempty so a tool-call-only assistant turn
	// serializes "content":null, which the API requires.
	Content    any        `json:"content"`
	ToolCalls  []toolCall `json:"tool_calls,omitempty"`   // assistant turns
	ToolCallID string     `json:"tool_call_id,omitempty"` // "tool" turns
	Name       string     `json:"name,omitempty"`
}

// contentPart is one element of a multimodal content array.
type contentPart struct {
	Type     string    `json:"type"` // "text" | "image_url"
	Text     string    `json:"text,omitempty"`
	ImageURL *imageURL `json:"image_url,omitempty"`
}

// imageURL carries an image as a data URL ("data:image/png;base64,...") or a
// regular http(s) URL.
type imageURL struct {
	URL    string `json:"url"`
	Detail string `json:"detail,omitempty"`
}

// tool describes a callable function in OpenAI's tools format.
type tool struct {
	Type     string       `json:"type"` // "function"
	Function toolFunction `json:"function"`
}

// toolFunction is the function schema inside a tool definition.
type toolFunction struct {
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	Parameters  json.RawMessage `json:"parameters,omitempty"`
}

// toolCall is a function call requested by the assistant (and echoed back in
// the assistant message that holds it).
type toolCall struct {
	Index    int          `json:"index,omitempty"` // present on streaming deltas
	ID       string       `json:"id,omitempty"`
	Type     string       `json:"type,omitempty"` // "function"
	Function functionCall `json:"function"`
}

// functionCall is the name + JSON-encoded arguments of a tool call.
type functionCall struct {
	Name string `json:"name,omitempty"`
	// Arguments is a JSON-encoded string (OpenAI's wire format), not an object.
	Arguments string `json:"arguments,omitempty"`
}

// streamOptions enables the trailing usage chunk during streaming.
type streamOptions struct {
	IncludeUsage bool `json:"include_usage"`
}

// ==========================================================================
// Response wire structs
// ==========================================================================

// chatCompletionResponse is the non-streaming /v1/chat/completions response.
type chatCompletionResponse struct {
	ID      string `json:"id"`
	Object  string `json:"object"`
	Created int64  `json:"created"`
	Model   string `json:"model"`
	Choices []struct {
		Index        int             `json:"index"`
		Message      responseMessage `json:"message"`
		FinishReason string          `json:"finish_reason"`
	} `json:"choices"`
	Usage usage `json:"usage"`
	// Error is populated when the API returns an error body with HTTP 200,
	// which some compatible servers do.
	Error *apiError `json:"error,omitempty"`
}

// responseMessage is the assistant message in a response choice. Content is
// always a string here (null deserializes to ""); image output is not part of
// chat completions.
type responseMessage struct {
	Role      string     `json:"role"`
	Content   string     `json:"content"`
	ToolCalls []toolCall `json:"tool_calls,omitempty"`
}

// usage is the token accounting block.
type usage struct {
	PromptTokens        int `json:"prompt_tokens"`
	CompletionTokens    int `json:"completion_tokens"`
	TotalTokens         int `json:"total_tokens"`
	PromptTokensDetails struct {
		CachedTokens int `json:"cached_tokens"`
	} `json:"prompt_tokens_details"`
}

// apiError is the error object returned by the API.
type apiError struct {
	Message string `json:"message"`
	Type    string `json:"type"`
	Code    string `json:"code"`
}

// streamChunk is one SSE "data:" payload during streaming.
type streamChunk struct {
	ID      string `json:"id"`
	Object  string `json:"object"`
	Created int64  `json:"created"`
	Model   string `json:"model"`
	Choices []struct {
		Index int `json:"index"`
		Delta struct {
			Role    string `json:"role"`
			Content string `json:"content"`
			// Reasoning models stream chain-of-thought in a separate field whose
			// name varies across OpenAI-compatible servers: DeepSeek and older vLLM
			// reasoning parsers use reasoning_content; vLLM's GLM parser (0.23+) and
			// OpenRouter use reasoning. Capture both; parseSSEStream coalesces them.
			ReasoningContent string     `json:"reasoning_content"`
			Reasoning        string     `json:"reasoning"`
			ToolCalls        []toolCall `json:"tool_calls,omitempty"`
		} `json:"delta"`
		FinishReason *string `json:"finish_reason"`
	} `json:"choices"`
	Usage *usage `json:"usage,omitempty"`
}
