# ace-llm-openai-compat

Compile-time OpenAI-compatible adapter for [Ace](https://github.com/aceobservability/ace).

Implements `github.com/aceobservability/ace/backend/pkg/llm` (`AIProvider`). `init` calls `llm.RegisterLLM` for `openai`, `openrouter`, `ollama`, and `custom`. Ace blank-imports this module.

```go
import _ "github.com/aceobservability/ace-llm-openai-compat"
```

List-models calls `GET /models`. Chat calls `POST /chat/completions` and copies JSON or SSE. Tests use `httptest` fixtures. No live OpenAI API in CI.
