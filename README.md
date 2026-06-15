# openai

A Go client for OpenAI's **Chat Completions** API (`/v1/chat/completions`) that
implements the [`github.com/wbrown/llmapi`](https://github.com/wbrown/llmapi)
`Conversation` interface. It is a sibling of
[`github.com/wbrown/anthropic`](https://github.com/wbrown/anthropic) and
[`github.com/wbrown/novelai`](https://github.com/wbrown/novelai), so code written
against `llmapi` can switch providers with no changes.

Because it speaks plain Chat Completions, `SetEndpoint` points it at any
OpenAI-compatible server — Azure OpenAI, vLLM, Together, Groq, OpenRouter, or a
local llama.cpp / Ollama endpoint. The endpoint is a **base URL** (the `/v1`
root, e.g. `http://host:8000/v1`); `/chat/completions` is appended per request.

## Usage

```go
conv := openai.NewConversation("You are a helpful assistant.")
conv.SetModel("gpt-4o") // required: there is no default model
reply, stopReason, inTok, outTok, _, cacheRead, err := conv.Send("Hello!", llmapi.Sampling{})
```

The API token is read from `OPENAI_API_KEY` (or `~/.openai_key`, `./.openai_key`)
into `DefaultApiToken`; set `conv.ApiToken` to override per conversation. **Auth
is optional** — when the token is empty, no `Authorization` header is sent, so
the client works against unauthenticated local servers (e.g. a default vLLM).

### Streaming

```go
reply, stop, _, _, _, _, err := conv.SendStreaming("Tell me a story.", llmapi.Sampling{},
    func(text string, done bool) { fmt.Print(text) })
```

### Tools and images

`SetTools` registers OpenAI function tools; `SendRich` accepts multimodal
content blocks (text + `image_url`). Tool calls in the response surface as
`ToolUse` blocks via `SendRich` / `GetRichMessages`.

## Behavior notes

- **No default model.** Model selection is the caller's decision; `Send` returns
  an error until you set `Settings.Model` (or call `SetModel`).
- **`max_completion_tokens`.** `Settings.MaxTokens` is sent as
  `max_completion_tokens` (the modern field; `max_tokens` is deprecated and
  rejected by reasoning models).
- **Sampling is omitted when zero.** Unset `Temperature`/`TopP` are omitted from
  the request so the server applies its default — which is what lets reasoning
  models (o-series, gpt-5) work without a per-model table. `top_k` is not a
  standard chat field and is not sent.
- **Caching is automatic.** OpenAI caches eligible prompts server-side with no
  client control, so `EnableSystemCaching` / `EnableConversationCaching` /
  `DisableConversationCaching` are no-ops returning `nil`. Cache hits surface as
  `cacheReadTokens` (from `usage.prompt_tokens_details.cached_tokens`);
  `cacheCreationTokens` is always 0.
- **Continuation.** `SendUntilDone` / `SendStreamingUntilDone` re-prompt with a
  `"Continue."` user message on `max_tokens`, since chat completions cannot
  continue an assistant turn.

## Testing

Tests run a real `Conversation` against a real, embedded inference server —
[`github.com/wbrown/tinyoai`](https://github.com/wbrown/tinyoai), a pure-Go
OpenAI-compatible server backed by a tiny model — plus a request-recording
middleware to assert the client's wire format. No mocks, no API key, no network.

```bash
go test ./...
```

## License

MIT.
