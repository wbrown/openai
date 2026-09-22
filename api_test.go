package openai

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
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
	cb := func(d llmapi.StreamDelta) {
		if d.Done {
			sawDone = true
			return
		}
		if d.Text != "" {
			chunks = append(chunks, d.Text)
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

// TestParseSSEStreamReasoningContent feeds the parser a crafted OpenAI-compatible
// SSE stream interleaving reasoning_content and content deltas — the wire shape a
// vLLM server launched with a reasoning parser (GLM/DeepSeek) emits. tinyoai is a
// real transformer with no reasoning concept, so this path cannot be exercised
// through it; parseSSEStream takes a plain io.Reader, so we drive it directly with
// the controlled wire bytes a reasoning server would send. Contract: reasoning is
// forwarded live through the callback (so it is visible), but is NOT part of the
// returned reply (the generated content), so it never lands in stored output.
//
// The finish reason is asserted as the raw "stop": parseSSEStream returns the
// server's finish_reason verbatim on the account's finishReason; the
// stop→end_turn mapping happens downstream when SendStreaming projects the
// account, which is why TestStreaming (calling the public SendStreaming) sees
// "end_turn" while this direct-parser test sees "stop".
func TestParseSSEStreamReasoningContent(t *testing.T) {
	// Distinct reasoning vs content strings so assertions can tell them apart.
	const reasoningA = "Let me reason about the request. "
	const reasoningB = "Two steps, then answer."
	const contentA = "The final "
	const contentB = "answer is here."

	chunk := func(delta string) string {
		return `data: {"choices":[{"index":0,"delta":` + delta + `}]}`
	}
	sse := strings.Join([]string{
		chunk(`{"role":"assistant"}`),
		chunk(`{"reasoning_content":"` + reasoningA + `"}`),
		chunk(`{"reasoning_content":"` + reasoningB + `"}`),
		chunk(`{"content":"` + contentA + `"}`),
		chunk(`{"content":"` + contentB + `"}`),
		`data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}`,
		"data: [DONE]",
		"",
	}, "\n\n")

	var reasoning, content strings.Builder
	var sawDone bool
	cb := func(d llmapi.StreamDelta) {
		if d.Done {
			sawDone = true
			return
		}
		switch d.Kind {
		case llmapi.TokenReasoning:
			reasoning.WriteString(d.Text)
		case llmapi.TokenContent:
			content.WriteString(d.Text)
		}
	}

	parsed, err := parseSSEStream(strings.NewReader(sse), cb)
	if err != nil {
		t.Fatalf("parseSSEStream: %v", err)
	}
	text, stopReason := parsed.text, parsed.finishReason

	// Reasoning deltas are tagged TokenReasoning and carry the chain-of-thought;
	// content deltas are tagged TokenContent and carry the answer.
	if got := reasoning.String(); got != reasoningA+reasoningB {
		t.Errorf("reasoning stream = %q, want %q", got, reasoningA+reasoningB)
	}
	if got := content.String(); got != contentA+contentB {
		t.Errorf("content stream = %q, want %q", got, contentA+contentB)
	}

	// The returned reply is the generated content ONLY — reasoning must not leak in.
	if want := contentA + contentB; text != want {
		t.Errorf("reply = %q, want %q (content only, no reasoning)", text, want)
	}

	if !sawDone {
		t.Error("callback never signaled done")
	}
	// Raw finish_reason from the parser; see the doc comment above.
	if stopReason != "stop" {
		t.Errorf("stop = %q, want %q", stopReason, "stop")
	}
}

// sseFixtureConversation serves one recorded SSE stream to every request and
// returns a Conversation pointed at it, plus the request recorder. The bytes
// are served as the server put them on the wire, so the account the client
// reports is checked against real chunk shapes.
func sseFixtureConversation(t *testing.T, sse []byte) (*Conversation, *recordingHandler) {
	t.Helper()
	replay := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		if _, err := w.Write(sse); err != nil {
			t.Errorf("serve fixture: %v", err)
		}
	})
	rec := &recordingHandler{inner: replay}
	srv := httptest.NewServer(rec)
	t.Cleanup(srv.Close)

	conv := NewConversation("")
	conv.SetEndpoint(srv.URL)
	conv.SetModel("reasoner")
	conv.ApiToken = "test-key"
	return conv, rec
}

// readFixture returns the bytes of one file under testdata.
func readFixture(t *testing.T, name string) []byte {
	t.Helper()
	data, err := os.ReadFile("testdata/" + name)
	if err != nil {
		t.Fatalf("read fixture %s: %v", name, err)
	}
	return data
}

// TestSendRichStreaming_AccountsForTheRequest drives a vLLM stream captured
// from a reasoning deployment (reasoning deltas, then content, finish_reason
// "stop", a trailing usage chunk, and per-chunk token_ids) through
// SendRichStreaming and asserts the response accounts for the request: the
// completion budget the request carried on the wire, the server's own finish
// reason beside the normalized stop, and the output tokens attributed per
// channel from the token ids, summing to the server's completion total.
func TestSendRichStreaming_AccountsForTheRequest(t *testing.T) {
	conv, rec := sseFixtureConversation(t, readFixture(t, "vllm_reasoning_then_content.sse"))
	conv.Settings.MaxTokens = 8192
	conv.Settings.OutputCeiling = 65536

	rr, err := conv.SendRichStreaming(
		[]llmapi.ContentBlock{llmapi.NewTextBlock("In one sentence, why does a lever with a longer arm lift a heavier load?")},
		llmapi.Sampling{ReasoningEffort: llmapi.ReasoningHigh}, nil)
	if err != nil {
		t.Fatalf("SendRichStreaming: %v", err)
	}

	wire := rec.lastRequest(t)["max_completion_tokens"].(float64)
	if rr.CompletionBudget != 65536 || int(wire) != rr.CompletionBudget {
		t.Errorf("CompletionBudget = %d, wire max_completion_tokens = %v; want both 65536 (reasoning on with the deployment ceiling known: the ceiling is the budget)", rr.CompletionBudget, wire)
	}
	if rr.FinishReason != "stop" {
		t.Errorf("FinishReason = %q, want %q (the server's own word)", rr.FinishReason, "stop")
	}
	if rr.StopReason != "end_turn" {
		t.Errorf("StopReason = %q, want %q", rr.StopReason, "end_turn")
	}
	if rr.InputTokens != 29 || rr.OutputTokens != 553 {
		t.Errorf("tokens in=%d out=%d, want in=29 out=553 (the usage chunk)", rr.InputTokens, rr.OutputTokens)
	}
	want := llmapi.OutputTokenSplit{Reasoning: 522, Content: 31, Known: true}
	if rr.OutputSplit != want {
		t.Errorf("OutputSplit = %+v, want %+v (per-chunk token_ids attributed to the channel of each chunk's delta)", rr.OutputSplit, want)
	}
	if got := rr.OutputSplit.Reasoning + rr.OutputSplit.Content; got != rr.OutputTokens {
		t.Errorf("split sums to %d, want the server's completion_tokens %d", got, rr.OutputTokens)
	}
	const answer = "A lever with a longer arm can lift a heavier load because the increased distance from the fulcrum generates greater torque, effectively multiplying the applied force."
	if rr.Text() != answer {
		t.Errorf("Text() = %q, want the content deltas only: %q", rr.Text(), answer)
	}
}

// TestSendRichStreaming_ReasoningOnlyLengthCut drives a stream that ends on
// finish_reason "length" after reasoning deltas only. The response reports
// the raw finish reason, an empty reply, and a known split with every output
// token on the reasoning channel.
func TestSendRichStreaming_ReasoningOnlyLengthCut(t *testing.T) {
	conv, _ := sseFixtureConversation(t, readFixture(t, "vllm_reasoning_only_length.sse"))
	conv.Settings.MaxTokens = 8192

	rr, err := conv.SendRichStreaming(
		[]llmapi.ContentBlock{llmapi.NewTextBlock("Lay out the plan.")},
		llmapi.Sampling{ReasoningEffort: llmapi.ReasoningHigh}, nil)
	if err != nil {
		t.Fatalf("SendRichStreaming: %v", err)
	}
	if rr.FinishReason != "length" {
		t.Errorf("FinishReason = %q, want %q", rr.FinishReason, "length")
	}
	if rr.StopReason != "max_tokens" {
		t.Errorf("StopReason = %q, want %q", rr.StopReason, "max_tokens")
	}
	if rr.Text() != "" {
		t.Errorf("Text() = %q, want empty: the stream carried no content delta", rr.Text())
	}
	if rr.CompletionBudget != 24576 {
		t.Errorf("CompletionBudget = %d, want 24576 (8192 desired + 16384 high headroom, no ceiling)", rr.CompletionBudget)
	}
	want := llmapi.OutputTokenSplit{Reasoning: 21, Content: 0, Known: true}
	if rr.OutputSplit != want {
		t.Errorf("OutputSplit = %+v, want %+v", rr.OutputSplit, want)
	}
	if rr.OutputTokens != 21 {
		t.Errorf("OutputTokens = %d, want 21", rr.OutputTokens)
	}
}

// TestSendRichStreaming_SplitWithoutTokenIDs covers servers that send no
// per-chunk token_ids. One that reports reasoning_tokens in the usage chunk's
// completion_tokens_details attributes the split through it, the content
// count being the remainder of completion_tokens. One that reports neither
// leaves the split unknown.
func TestSendRichStreaming_SplitWithoutTokenIDs(t *testing.T) {
	chunk := func(delta string) string {
		return `data: {"choices":[{"index":0,"delta":` + delta + `,"finish_reason":null}]}`
	}
	stream := func(usage string) []byte {
		return []byte(strings.Join([]string{
			chunk(`{"role":"assistant","content":""}`),
			chunk(`{"reasoning_content":"Weighing the two readings. "}`),
			chunk(`{"reasoning_content":"The second holds."}`),
			chunk(`{"content":"The second "}`),
			chunk(`{"content":"reading holds."}`),
			`data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}`,
			`data: {"choices":[],"usage":` + usage + `}`,
			"data: [DONE]",
			"",
		}, "\n\n"))
	}

	t.Run("from usage details", func(t *testing.T) {
		conv, _ := sseFixtureConversation(t, stream(`{"prompt_tokens":18,"completion_tokens":12,"total_tokens":30,"completion_tokens_details":{"reasoning_tokens":7}}`))
		rr, err := conv.SendRichStreaming([]llmapi.ContentBlock{llmapi.NewTextBlock("Which reading holds?")}, llmapi.Sampling{ReasoningEffort: llmapi.ReasoningLow}, nil)
		if err != nil {
			t.Fatalf("SendRichStreaming: %v", err)
		}
		want := llmapi.OutputTokenSplit{Reasoning: 7, Content: 5, Known: true}
		if rr.OutputSplit != want {
			t.Errorf("OutputSplit = %+v, want %+v (reasoning_tokens from usage details, content the remainder)", rr.OutputSplit, want)
		}
	})

	t.Run("unknown when the server attributes nothing", func(t *testing.T) {
		conv, _ := sseFixtureConversation(t, stream(`{"prompt_tokens":18,"completion_tokens":12,"total_tokens":30}`))
		rr, err := conv.SendRichStreaming([]llmapi.ContentBlock{llmapi.NewTextBlock("Which reading holds?")}, llmapi.Sampling{ReasoningEffort: llmapi.ReasoningLow}, nil)
		if err != nil {
			t.Fatalf("SendRichStreaming: %v", err)
		}
		if rr.OutputSplit.Known {
			t.Errorf("OutputSplit = %+v, want unknown: the server reported only a completion total", rr.OutputSplit)
		}
		if rr.OutputTokens != 12 {
			t.Errorf("OutputTokens = %d, want 12", rr.OutputTokens)
		}
	})
}

// TestSendRich_AccountsForTheRequest checks the non-streaming path reports the
// same account: the budget the request carried and the server's own finish
// reason paired with its normalized stop.
func TestSendRich_AccountsForTheRequest(t *testing.T) {
	conv, rec := newConversation(t, "")
	conv.Settings.MaxTokens = 12

	rr, err := conv.SendRich([]llmapi.ContentBlock{llmapi.NewTextBlock("Once upon a time")}, llmapi.Sampling{})
	if err != nil {
		t.Fatalf("SendRich: %v", err)
	}
	wire := rec.lastRequest(t)["max_completion_tokens"].(float64)
	if rr.CompletionBudget != 12 || int(wire) != rr.CompletionBudget {
		t.Errorf("CompletionBudget = %d, wire max_completion_tokens = %v; want both 12", rr.CompletionBudget, wire)
	}
	pairs := map[string]string{"stop": "end_turn", "length": "max_tokens"}
	if want, ok := pairs[rr.FinishReason]; !ok || rr.StopReason != want {
		t.Errorf("FinishReason = %q, StopReason = %q; want the server's stop or length paired with its normalized form", rr.FinishReason, rr.StopReason)
	}
}

// TestReasoningEffortChatTemplateKwargs pins the per-call reasoning-effort mapping
// into vLLM chat_template_kwargs: ReasoningOff -> {"enable_thinking": false}, every
// other level -> {"thinking": true, "enable_thinking": true, "reasoning_effort":
// "<level>"}. All three on-conventions are sent explicitly so a template keyed off
// any one of them (including Qwen3-style enable_thinking, which some chat templates
// treat as off when absent rather than defaulting to on) actually reasons. The zero
// value (ReasoningOff) means a bare Sampling{} disables reasoning by default.
func TestReasoningEffortChatTemplateKwargs(t *testing.T) {
	levels := []llmapi.ReasoningEffort{llmapi.ReasoningLow, llmapi.ReasoningMedium, llmapi.ReasoningHigh, llmapi.ReasoningXHigh, llmapi.ReasoningMax}

	// ReasoningOff: only enable_thinking:false, nothing else.
	{
		conv := NewConversation("")
		conv.SetModel("test-model")
		conv.AddMessage(llmapi.RoleUser, "hi")
		req, err := conv.buildRequest(llmapi.Sampling{ReasoningEffort: llmapi.ReasoningOff}, false)
		if err != nil {
			t.Fatalf("effort off: buildRequest: %v", err)
		}
		if got, ok := req.ChatTemplateKwargs["enable_thinking"]; !ok || got != false {
			t.Errorf("off: chat_template_kwargs[enable_thinking] = %v (present=%v), want false", got, ok)
		}
		if _, ok := req.ChatTemplateKwargs["reasoning_effort"]; ok {
			t.Error("off must not set reasoning_effort")
		}
		if _, ok := req.ChatTemplateKwargs["thinking"]; ok {
			t.Error("off must not set thinking")
		}
	}

	// Every non-off level: all three on-conventions present.
	for _, effort := range levels {
		conv := NewConversation("")
		conv.SetModel("test-model")
		conv.AddMessage(llmapi.RoleUser, "hi")
		req, err := conv.buildRequest(llmapi.Sampling{ReasoningEffort: effort}, false)
		if err != nil {
			t.Fatalf("effort %v: buildRequest: %v", effort, err)
		}
		if got, ok := req.ChatTemplateKwargs["thinking"]; !ok || got != true {
			t.Errorf("effort %v: chat_template_kwargs[thinking] = %v (present=%v), want true", effort, got, ok)
		}
		if got, ok := req.ChatTemplateKwargs["enable_thinking"]; !ok || got != true {
			t.Errorf("effort %v: chat_template_kwargs[enable_thinking] = %v (present=%v), want true", effort, got, ok)
		}
		if got, ok := req.ChatTemplateKwargs["reasoning_effort"]; !ok || got != effort.String() {
			t.Errorf("effort %v: chat_template_kwargs[reasoning_effort] = %v (present=%v), want %v", effort, got, ok, effort.String())
		}
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

// TestWireBudgetComputation pins the wire max_completion_tokens computation.
// Reasoning models emit thinking into the same completion budget as the
// answer, and how much they think is not a function of the effort tier, so a
// reasoning-on request's budget is Settings.OutputCeiling, the deployment's
// real per-request output limit, whenever the deployment reports one. Where
// the ceiling is unknown (0), the budget is the desired output (per-call
// Sampling.DesiredOutputTokens, else Settings.MaxTokens as the default desired
// output) plus the effort tier's reasoning headroom. Reasoning off sends the
// desired output, clamped to the ceiling. With no desired output at all, the
// field stays omitted and the server's own default governs; headroom is never
// added to a bound the caller declined to set.
func TestWireBudgetComputation(t *testing.T) {
	send := func(t *testing.T, mutate func(*Conversation), sampling llmapi.Sampling) map[string]any {
		t.Helper()
		conv, rec := newConversation(t, "")
		conv.Settings.StopSequences = []string{"."} // keep the real inference short
		mutate(conv)
		if _, _, _, _, _, _, err := conv.Send("Once upon a time", sampling); err != nil {
			t.Fatalf("Send: %v", err)
		}
		return rec.lastRequest(t)
	}

	t.Run("reasoning effort adds headroom to the settings default", func(t *testing.T) {
		body := send(t, func(c *Conversation) { c.Settings.MaxTokens = 2048 },
			llmapi.Sampling{ReasoningEffort: llmapi.ReasoningHigh})
		if v := body["max_completion_tokens"].(float64); v != 18432 {
			t.Errorf("max_completion_tokens = %v, want 18432 (2048 desired + 16384 high headroom)", v)
		}
	})

	t.Run("per-call DesiredOutputTokens overrides the settings default", func(t *testing.T) {
		body := send(t, func(c *Conversation) { c.Settings.MaxTokens = 2048 },
			llmapi.Sampling{ReasoningEffort: llmapi.ReasoningHigh, DesiredOutputTokens: 4096})
		if v := body["max_completion_tokens"].(float64); v != 20480 {
			t.Errorf("max_completion_tokens = %v, want 20480 (4096 desired + 16384 high headroom)", v)
		}
	})

	t.Run("off reserves nothing", func(t *testing.T) {
		body := send(t, func(c *Conversation) { c.Settings.MaxTokens = 2048 },
			llmapi.Sampling{DesiredOutputTokens: 1024})
		if v := body["max_completion_tokens"].(float64); v != 1024 {
			t.Errorf("max_completion_tokens = %v, want 1024 (desired only; reasoning off)", v)
		}
	})

	t.Run("OutputCeiling clamps the wire total", func(t *testing.T) {
		body := send(t, func(c *Conversation) {
			c.Settings.MaxTokens = 4096
			c.Settings.OutputCeiling = 8192
		}, llmapi.Sampling{ReasoningEffort: llmapi.ReasoningHigh})
		if v := body["max_completion_tokens"].(float64); v != 8192 {
			t.Errorf("max_completion_tokens = %v, want 8192 (4096+16384 clamped to the deployment ceiling)", v)
		}
	})

	t.Run("reasoning on takes a known deployment ceiling as its budget", func(t *testing.T) {
		body := send(t, func(c *Conversation) {
			c.Settings.MaxTokens = 8192
			c.Settings.OutputCeiling = 65536
		}, llmapi.Sampling{ReasoningEffort: llmapi.ReasoningHigh})
		if v := body["max_completion_tokens"].(float64); v != 65536 {
			t.Errorf("max_completion_tokens = %v, want 65536 (the deployment ceiling: how much the model thinks is not a function of the effort tier)", v)
		}
	})

	t.Run("reasoning off with a known ceiling sends the desired output", func(t *testing.T) {
		body := send(t, func(c *Conversation) {
			c.Settings.MaxTokens = 8192
			c.Settings.OutputCeiling = 65536
		}, llmapi.Sampling{})
		if v := body["max_completion_tokens"].(float64); v != 8192 {
			t.Errorf("max_completion_tokens = %v, want 8192 (desired only; reasoning off)", v)
		}
	})

	t.Run("no desired output leaves the field omitted even with effort", func(t *testing.T) {
		body := send(t, func(c *Conversation) {
			c.Settings.MaxTokens = 0
			c.Settings.StopSequences = []string{" "} // unbounded request; stop fast
		}, llmapi.Sampling{ReasoningEffort: llmapi.ReasoningHigh})
		if v, ok := body["max_completion_tokens"]; ok {
			t.Errorf("max_completion_tokens = %v, want omitted (caller set no output bound; headroom must not fabricate one)", v)
		}
	})
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
