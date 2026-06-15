package openai

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"

	"github.com/wbrown/llmapi"
	"github.com/wbrown/tinyoai"
)

// recordingHandler captures the most recent request body, then delegates to the
// real tinyoai inference server. It is test infrastructure for asserting the
// client's wire format; the production server carries no such state.
type recordingHandler struct {
	inner      http.Handler
	mu         sync.Mutex
	lastBody   []byte
	lastHeader http.Header
}

func (h *recordingHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	body, _ := io.ReadAll(r.Body)
	h.mu.Lock()
	h.lastBody = body
	h.lastHeader = r.Header.Clone()
	h.mu.Unlock()
	r.Body = io.NopCloser(bytes.NewReader(body))
	h.inner.ServeHTTP(w, r)
}

// lastAuth returns the Authorization header captured from the most recent request.
func (h *recordingHandler) lastAuth() string {
	h.mu.Lock()
	defer h.mu.Unlock()
	return h.lastHeader.Get("Authorization")
}

// lastRequest decodes the most recently captured request body.
func (h *recordingHandler) lastRequest(t *testing.T) map[string]any {
	t.Helper()
	h.mu.Lock()
	defer h.mu.Unlock()
	var m map[string]any
	if err := json.Unmarshal(h.lastBody, &m); err != nil {
		t.Fatalf("decode captured request: %v", err)
	}
	return m
}

// newConversation starts a tinyoai-backed server and returns a Conversation
// pointed at it (model and token set), plus the request recorder.
func newConversation(t *testing.T, system string) (*Conversation, *recordingHandler) {
	t.Helper()
	backend, err := tinyoai.NewDefaultServer()
	if err != nil {
		t.Fatalf("tinyoai server: %v", err)
	}
	rec := &recordingHandler{inner: backend}
	srv := httptest.NewServer(rec)
	t.Cleanup(srv.Close)

	conv := NewConversation(system)
	conv.SetEndpoint(srv.URL) // base URL; the client appends /chat/completions
	conv.SetModel("stories260K")
	conv.ApiToken = "test-key" // tinyoai ignores auth; a non-empty token exercises the Authorization path
	return conv, rec
}

func TestNewConversation(t *testing.T) {
	conv := NewConversation("be brief")
	if conv.GetSystem() != "be brief" {
		t.Errorf("system = %q", conv.GetSystem())
	}
	if conv.Settings.Model != "" {
		t.Errorf("model should be unset by default, got %q", conv.Settings.Model)
	}
	if len(conv.Messages) != 0 {
		t.Errorf("expected empty history, got %d", len(conv.Messages))
	}
	if conv.HttpClient == nil {
		t.Error("HttpClient not initialized")
	}
}

func TestSendRequiresModel(t *testing.T) {
	conv := NewConversation("sys")
	conv.ApiToken = "k"
	_, _, _, _, _, _, err := conv.Send("hi", llmapi.Sampling{})
	if err == nil || !strings.Contains(err.Error(), "model not set") {
		t.Fatalf("expected 'model not set' error, got %v", err)
	}
}

func TestSend(t *testing.T) {
	conv, _ := newConversation(t, "You are a storyteller.")
	reply, stop, in, out, cacheCreate, cacheRead, err := conv.Send("Once upon a time", llmapi.Sampling{})
	if err != nil {
		t.Fatalf("Send: %v", err)
	}
	t.Logf("reply=%q stop=%s in=%d out=%d cacheRead=%d", reply, stop, in, out, cacheRead)
	if strings.TrimSpace(reply) == "" {
		t.Error("empty reply")
	}
	if stop != "end_turn" && stop != "max_tokens" {
		t.Errorf("unexpected stop reason %q", stop)
	}
	if in == 0 || out == 0 {
		t.Errorf("expected non-zero tokens, got in=%d out=%d", in, out)
	}
	if cacheCreate != 0 {
		t.Errorf("cacheCreate should be 0 for OpenAI, got %d", cacheCreate)
	}
	if len(conv.Messages) != 2 {
		t.Errorf("expected 2 messages, got %d", len(conv.Messages))
	}
}

func TestSendMaxTokensFinish(t *testing.T) {
	conv, _ := newConversation(t, "")
	conv.Settings.MaxTokens = 4
	_, stop, _, out, _, _, err := conv.Send("The dog", llmapi.Sampling{})
	if err != nil {
		t.Fatalf("Send: %v", err)
	}
	if stop != "max_tokens" {
		t.Errorf("stop = %q, want max_tokens", stop)
	}
	if out == 0 || out > 4 {
		t.Errorf("completion tokens = %d, want 1..4", out)
	}
}

func TestMultiTurnSystemPrependedOnce(t *testing.T) {
	conv, rec := newConversation(t, "You are a bot.")
	conv.Settings.MaxTokens = 6

	if _, _, _, _, _, _, err := conv.Send("Once upon a time", llmapi.Sampling{}); err != nil {
		t.Fatalf("turn 1: %v", err)
	}
	if _, _, _, _, _, _, err := conv.Send("the dog ran", llmapi.Sampling{}); err != nil {
		t.Fatalf("turn 2: %v", err)
	}
	if len(conv.Messages) != 4 {
		t.Fatalf("expected 4 messages, got %d", len(conv.Messages))
	}

	// The second request must carry the system message exactly once, at index 0.
	msgs := requestMessagesField(t, rec.lastRequest(t))
	systemCount := 0
	for i, raw := range msgs {
		msg := raw.(map[string]any)
		if msg["role"] == "system" {
			systemCount++
			if i != 0 {
				t.Errorf("system message at index %d, want 0", i)
			}
		}
	}
	if systemCount != 1 {
		t.Errorf("system message appears %d times, want 1", systemCount)
	}
}

func TestStreaming(t *testing.T) {
	conv, _ := newConversation(t, "")
	conv.Settings.MaxTokens = 16

	var chunks []string
	var sawDone bool
	cb := func(text string, done bool) {
		if done {
			sawDone = true
			return
		}
		if text != "" {
			chunks = append(chunks, text)
		}
	}

	reply, stop, in, out, _, _, err := conv.SendStreaming("Once upon a time", llmapi.Sampling{}, cb)
	if err != nil {
		t.Fatalf("SendStreaming: %v", err)
	}
	t.Logf("reply=%q stop=%s chunks=%d", reply, stop, len(chunks))
	if strings.TrimSpace(reply) == "" {
		t.Error("empty reply")
	}
	if len(chunks) == 0 {
		t.Error("no streamed chunks")
	}
	if got := strings.Join(chunks, ""); got != reply {
		t.Errorf("chunks %q != reply %q", got, reply)
	}
	if !sawDone {
		t.Error("callback never signaled done")
	}
	if in == 0 || out == 0 {
		t.Errorf("expected non-zero tokens, got in=%d out=%d", in, out)
	}
	if stop != "end_turn" && stop != "max_tokens" {
		t.Errorf("unexpected stop %q", stop)
	}
}

// TestSendUntilDoneContinues exercises the real continuation loop. MaxTokens=1
// forces the max_tokens branch on every call, so the loop must continue, and a
// fixed seed makes the multi-call sequence deterministic. The loop ends when the
// model returns a non-max_tokens stop — here its end-of-story token, with the
// "." stop sequence as a backstop bound; both normalize to "end_turn". The
// continuation branch is proven by the "Continue." user messages the loop
// appends before each follow-up call.
func TestSendUntilDoneContinues(t *testing.T) {
	conv, _ := newConversation(t, "")
	conv.Settings.MaxTokens = 1
	conv.Settings.Seed = 42
	conv.Settings.StopSequences = []string{"."}

	reply, stop, _, out, _, _, err := conv.SendUntilDone("Once upon a time", llmapi.Sampling{})
	if err != nil {
		t.Fatalf("SendUntilDone: %v", err)
	}
	t.Logf("reply=%q stop=%s out=%d", reply, stop, out)

	// The continuation branch must have run: each follow-up call appends "Continue.".
	continues := 0
	for _, m := range conv.Messages {
		if m.Role == "user" && contentString(m.Content) == "Continue." {
			continues++
		}
	}
	if continues == 0 {
		t.Error("loop never continued: no 'Continue.' message in history")
	}
	// The loop must terminate on a real stop, not by exhausting max_tokens.
	if stop != "end_turn" {
		t.Errorf("stop = %q, want end_turn (loop must run until a non-max_tokens stop)", stop)
	}
	if strings.TrimSpace(reply) == "" {
		t.Error("expected a non-empty accumulated reply")
	}
	t.Logf("continuation iterations: %d", continues)
}

func TestSendsMaxCompletionTokens(t *testing.T) {
	conv, rec := newConversation(t, "")
	conv.Settings.MaxTokens = 12
	if _, _, _, _, _, _, err := conv.Send("Once upon a time", llmapi.Sampling{}); err != nil {
		t.Fatalf("Send: %v", err)
	}
	body := rec.lastRequest(t)
	if _, ok := body["max_tokens"]; ok {
		t.Error("request used deprecated max_tokens")
	}
	v, ok := body["max_completion_tokens"]
	if !ok {
		t.Fatal("request missing max_completion_tokens")
	}
	if v.(float64) != 12 {
		t.Errorf("max_completion_tokens = %v, want 12", v)
	}
}

func TestSerializesTools(t *testing.T) {
	conv, rec := newConversation(t, "")
	conv.Settings.MaxTokens = 4
	conv.SetTools([]llmapi.ToolDefinition{{
		Name:        "get_weather",
		Description: "Get the weather",
		InputSchema: json.RawMessage(`{"type":"object","properties":{"city":{"type":"string"}}}`),
	}})
	if _, _, _, _, _, _, err := conv.Send("Once upon a time", llmapi.Sampling{}); err != nil {
		t.Fatalf("Send: %v", err)
	}

	body := rec.lastRequest(t)
	tools, ok := body["tools"].([]any)
	if !ok || len(tools) != 1 {
		t.Fatalf("tools = %v", body["tools"])
	}
	tool := tools[0].(map[string]any)
	if tool["type"] != "function" {
		t.Errorf("tool type = %v, want function", tool["type"])
	}
	fn := tool["function"].(map[string]any)
	if fn["name"] != "get_weather" {
		t.Errorf("function name = %v", fn["name"])
	}
	if fn["parameters"] == nil {
		t.Error("function parameters missing")
	}
}

func TestSerializesImageBlock(t *testing.T) {
	conv, rec := newConversation(t, "")
	conv.Settings.MaxTokens = 4
	_, err := conv.SendRich([]llmapi.ContentBlock{
		llmapi.NewTextBlock("describe this"),
		llmapi.NewImageBlock(llmapi.MediaTypePNG, "aGVsbG8="),
	}, llmapi.Sampling{})
	if err != nil {
		t.Fatalf("SendRich: %v", err)
	}

	msgs := requestMessagesField(t, rec.lastRequest(t))
	last := msgs[len(msgs)-1].(map[string]any)
	parts, ok := last["content"].([]any)
	if !ok {
		t.Fatalf("expected array content, got %T", last["content"])
	}
	var foundImage bool
	for _, p := range parts {
		part := p.(map[string]any)
		if part["type"] == "image_url" {
			foundImage = true
			url := part["image_url"].(map[string]any)["url"].(string)
			if !strings.HasPrefix(url, "data:image/png;base64,aGVsbG8=") {
				t.Errorf("image url = %q", url)
			}
		}
	}
	if !foundImage {
		t.Error("no image_url content part in request")
	}
}

func TestNormalizeFinishReason(t *testing.T) {
	cases := map[string]string{
		"stop":           "end_turn",
		"length":         "max_tokens",
		"tool_calls":     "tool_use",
		"function_call":  "tool_use",
		"content_filter": "content_filter",
		"":               "",
	}
	for in, want := range cases {
		if got := normalizeFinishReason(in); got != want {
			t.Errorf("normalizeFinishReason(%q) = %q, want %q", in, got, want)
		}
	}
}

func TestMessageToBlocksWithToolCall(t *testing.T) {
	blocks := messageToBlocks(responseMessage{
		Content: "let me check",
		ToolCalls: []toolCall{{
			ID:       "call_1",
			Type:     "function",
			Function: functionCall{Name: "lookup", Arguments: `{"q":"x"}`},
		}},
	})
	if len(blocks) != 2 {
		t.Fatalf("expected 2 blocks, got %d", len(blocks))
	}
	if blocks[0].Type != llmapi.ContentTypeText || blocks[0].Text != "let me check" {
		t.Errorf("block 0 = %+v", blocks[0])
	}
	if blocks[1].Type != llmapi.ContentTypeToolUse || blocks[1].ToolUse == nil {
		t.Fatalf("block 1 = %+v", blocks[1])
	}
	tu := blocks[1].ToolUse
	if tu.ID != "call_1" || tu.Name != "lookup" || string(tu.Input) != `{"q":"x"}` {
		t.Errorf("tool use = %+v (input %s)", tu, tu.Input)
	}
}

func TestToolResultRoundTrip(t *testing.T) {
	conv := NewConversation("")
	// Assistant requests a tool, we add a tool result, then inspect both the
	// stored wire form and the rich round-trip.
	conv.AddRichMessage(llmapi.RoleAssistant, []llmapi.ContentBlock{{
		Type:    llmapi.ContentTypeToolUse,
		ToolUse: &llmapi.ToolUseContent{ID: "call_1", Name: "lookup", Input: json.RawMessage(`{}`)},
	}})
	conv.AddRichMessage(llmapi.RoleUser, []llmapi.ContentBlock{
		llmapi.NewToolResultBlock("call_1", "sunny", false),
	})

	// The tool result must be stored as a role:"tool" message with tool_call_id.
	toolMsg := conv.Messages[len(conv.Messages)-1]
	if toolMsg.Role != "tool" || toolMsg.ToolCallID != "call_1" {
		t.Fatalf("tool message = %+v", toolMsg)
	}
	if got := contentString(toolMsg.Content); got != "sunny" {
		t.Errorf("tool content = %q", got)
	}

	// GetRichMessages maps it back to a user message carrying a tool_result block.
	rich := conv.GetRichMessages()
	last := rich[len(rich)-1]
	if last.Role != llmapi.RoleUser || len(last.Content) != 1 {
		t.Fatalf("rich tool result = %+v", last)
	}
	tr := last.Content[0].ToolResult
	if tr == nil || tr.ToolUseID != "call_1" || tr.Content != "sunny" {
		t.Errorf("tool result block = %+v", last.Content[0])
	}
}

func TestMergeIfLastTwoAssistant(t *testing.T) {
	conv := NewConversation("")
	conv.AddMessage(llmapi.RoleUser, "go")
	conv.AddMessage(llmapi.RoleAssistant, "part one ")
	conv.AddMessage(llmapi.RoleAssistant, "part two")
	conv.MergeIfLastTwoAssistant()

	if len(conv.Messages) != 2 {
		t.Fatalf("expected 2 messages after merge, got %d", len(conv.Messages))
	}
	if got := contentString(conv.Messages[1].Content); got != "part onepart two" {
		t.Errorf("merged content = %q", got)
	}
}

func TestMergeSkipsToolCallTurns(t *testing.T) {
	conv := NewConversation("")
	conv.AddMessage(llmapi.RoleAssistant, "text")
	conv.Messages = append(conv.Messages, chatMessage{
		Role:      "assistant",
		ToolCalls: []toolCall{{ID: "c1", Type: "function", Function: functionCall{Name: "f"}}},
	})
	conv.MergeIfLastTwoAssistant()
	if len(conv.Messages) != 2 {
		t.Errorf("tool-call assistant turn must not be merged, got %d messages", len(conv.Messages))
	}
}

func TestCapabilities(t *testing.T) {
	conv := NewConversation("")
	caps := conv.GetCapabilities()
	if !caps.SupportsImages || !caps.SupportsToolUse || !caps.SupportsStreaming || !caps.SupportsCaching {
		t.Errorf("unexpected capabilities: %+v", caps)
	}
	if caps.SupportsThinking || caps.SupportsDocuments {
		t.Errorf("chat completions does not support thinking/documents: %+v", caps)
	}
}

func TestCachingEnableIsNoop(t *testing.T) {
	conv := NewConversation("")
	if err := conv.EnableSystemCaching(); err != nil {
		t.Errorf("EnableSystemCaching: %v", err)
	}
	if err := conv.EnableConversationCaching(); err != nil {
		t.Errorf("EnableConversationCaching: %v", err)
	}
	if err := conv.DisableConversationCaching(); err != nil {
		t.Errorf("DisableConversationCaching: %v", err)
	}
}

func TestAuthHeaderOptional(t *testing.T) {
	t.Run("with token", func(t *testing.T) {
		conv, rec := newConversation(t, "")
		conv.ApiToken = "secret-key"
		conv.Settings.MaxTokens = 4
		if _, _, _, _, _, _, err := conv.Send("Once upon a time", llmapi.Sampling{}); err != nil {
			t.Fatalf("Send: %v", err)
		}
		if got := rec.lastAuth(); got != "Bearer secret-key" {
			t.Errorf("Authorization = %q, want %q", got, "Bearer secret-key")
		}
	})
	t.Run("without token", func(t *testing.T) {
		conv, rec := newConversation(t, "")
		conv.ApiToken = "" // no key, like a local vLLM endpoint
		conv.Settings.MaxTokens = 4
		if _, _, _, _, _, _, err := conv.Send("Once upon a time", llmapi.Sampling{}); err != nil {
			t.Fatalf("Send with no token must succeed: %v", err)
		}
		if got := rec.lastAuth(); got != "" {
			t.Errorf("Authorization should be absent for a tokenless conversation, got %q", got)
		}
	})
}

// requestMessagesField extracts the messages array from a decoded request body.
func requestMessagesField(t *testing.T, body map[string]any) []any {
	t.Helper()
	msgs, ok := body["messages"].([]any)
	if !ok {
		t.Fatalf("request has no messages array: %v", body["messages"])
	}
	return msgs
}
