// Command wire streams a chat-completions request against an OpenAI-compatible
// endpoint and prints every raw SSE line as it arrives, timestamped with the
// elapsed time since the request and the gap since the previous line. It does no
// parsing: use it to see what a server (for example a vLLM-served reasoning
// model) actually puts on the wire and WHEN — time-to-first-token, mid-stream
// gaps, and which delta field carries reasoning (reasoning_content vs content).
//
// Example (a vLLM reasoning endpoint, reasoning on):
//
//	go run ./cmd/wire \
//	  -endpoint http://host:30000/v1 \
//	  -model your-org/Your-Model-FP8 \
//	  -reasoning high -max-tokens 32768 \
//	  -prompt "Write three sentences about a lighthouse."
package main

import (
	"flag"
	"fmt"
	"os"

	"github.com/wbrown/llmapi"
	"github.com/wbrown/openai"
)

func main() {
	endpoint := flag.String("endpoint", "", "base URL ending in /v1 (e.g. http://host:port/v1); empty uses the OpenAI default")
	model := flag.String("model", "", "model id (required, e.g. your-org/Your-Model-FP8)")
	system := flag.String("system", "", "system prompt")
	prompt := flag.String("prompt", "", "user prompt (or use -prompt-file)")
	promptFile := flag.String("prompt-file", "", "read the user prompt from this file")
	reasoning := flag.String("reasoning", "high", "reasoning effort: off|low|medium|high|max")
	maxTokens := flag.Int("max-tokens", 32768, "max_completion_tokens")
	temperature := flag.Float64("temperature", 0, "temperature (0 = server/conversation default)")
	token := flag.String("token", "", "API bearer token (default: OPENAI_API_KEY or token file; many vLLM servers need none)")
	flag.Parse()

	if *model == "" {
		fmt.Fprintln(os.Stderr, "error: -model is required")
		os.Exit(1)
	}

	userPrompt := *prompt
	if *promptFile != "" {
		data, err := os.ReadFile(*promptFile)
		if err != nil {
			fmt.Fprintf(os.Stderr, "error reading -prompt-file: %v\n", err)
			os.Exit(1)
		}
		userPrompt = string(data)
	}
	if userPrompt == "" {
		fmt.Fprintln(os.Stderr, "error: -prompt or -prompt-file is required")
		os.Exit(1)
	}

	effort, err := llmapi.ParseReasoningEffort(*reasoning)
	if err != nil {
		fmt.Fprintf(os.Stderr, "error: %v\n", err)
		os.Exit(1)
	}

	conv := openai.NewConversation(*system)
	conv.SetModel(*model)
	conv.Settings.MaxTokens = *maxTokens
	if *temperature != 0 {
		conv.Settings.Temperature = *temperature
	}
	if *endpoint != "" {
		conv.SetEndpoint(*endpoint)
	}
	if *token != "" {
		conv.ApiToken = *token
	}

	sampling := llmapi.Sampling{
		ReasoningEffort: effort,
		Temperature:     *temperature,
	}

	if err := conv.StreamRaw(userPrompt, sampling, os.Stdout); err != nil {
		fmt.Fprintf(os.Stderr, "\nstream error: %v\n", err)
		os.Exit(1)
	}
}
