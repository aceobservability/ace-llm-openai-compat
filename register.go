package openaicompat

import "github.com/aceobservability/ace/backend/pkg/llm"

func init() {
	llm.RegisterLLM("openai", New)
	llm.RegisterLLM("openrouter", New)
	llm.RegisterLLM("ollama", New)
	llm.RegisterLLM("custom", New)
}
