package openaicompat

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/aceobservability/ace/backend/pkg/llm"
)

func TestListModels(t *testing.T) {
	mockServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/models" {
			t.Errorf("expected path /models, got %s", r.URL.Path)
		}
		if r.Method != http.MethodGet {
			t.Errorf("expected GET, got %s", r.Method)
		}
		auth := r.Header.Get("Authorization")
		if auth != "Bearer test-api-key" {
			t.Errorf("expected Authorization 'Bearer test-api-key', got '%s'", auth)
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]interface{}{
			"data": []map[string]interface{}{
				{
					"id":       "gpt-4o",
					"object":   "model",
					"owned_by": "openai",
				},
				{
					"id":       "gpt-4o-mini",
					"object":   "model",
					"owned_by": "openai",
				},
				{
					"id":       "llama-3.1-70b",
					"object":   "model",
					"owned_by": "meta",
				},
			},
		})
	}))
	defer mockServer.Close()

	provider, err := New(llm.LLMConfig{
		BaseURL:     mockServer.URL,
		APIKey:      "test-api-key",
		DisplayName: "TestProvider",
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	models, err := provider.ListModels(context.Background())
	if err != nil {
		t.Fatalf("ListModels returned error: %v", err)
	}

	if len(models) != 3 {
		t.Fatalf("expected 3 models, got %d", len(models))
	}

	if models[0].ID != "gpt-4o" {
		t.Errorf("expected model ID 'gpt-4o', got '%s'", models[0].ID)
	}
	if models[0].Name != "gpt-4o" {
		t.Errorf("expected model Name 'gpt-4o', got '%s'", models[0].Name)
	}
	if models[0].Vendor != "openai" {
		t.Errorf("expected model Vendor 'openai', got '%s'", models[0].Vendor)
	}

	if models[2].Vendor != "meta" {
		t.Errorf("expected model Vendor 'meta', got '%s'", models[2].Vendor)
	}
}

func TestListModels_NoAPIKey(t *testing.T) {
	mockServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		auth := r.Header.Get("Authorization")
		if auth != "" {
			t.Errorf("expected no Authorization header for keyless provider, got '%s'", auth)
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]interface{}{
			"data": []map[string]interface{}{
				{
					"id":       "llama3:latest",
					"object":   "model",
					"owned_by": "library",
				},
			},
		})
	}))
	defer mockServer.Close()

	provider, err := New(llm.LLMConfig{
		BaseURL:     mockServer.URL,
		DisplayName: "Ollama",
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	models, err := provider.ListModels(context.Background())
	if err != nil {
		t.Fatalf("ListModels returned error: %v", err)
	}

	if len(models) != 1 {
		t.Fatalf("expected 1 model, got %d", len(models))
	}
	if models[0].ID != "llama3:latest" {
		t.Errorf("expected model ID 'llama3:latest', got '%s'", models[0].ID)
	}
}

func TestListModels_ServerError(t *testing.T) {
	mockServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
		w.Write([]byte(`{"error": "internal server error"}`))
	}))
	defer mockServer.Close()

	provider, err := New(llm.LLMConfig{
		BaseURL:     mockServer.URL,
		APIKey:      "test-key",
		DisplayName: "TestProvider",
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	_, err = provider.ListModels(context.Background())
	if err == nil {
		t.Fatal("expected error on server 500, got nil")
	}
	if !strings.Contains(err.Error(), "500") {
		t.Errorf("expected error to contain '500', got: %s", err.Error())
	}
}

func TestChat_NonStreaming(t *testing.T) {
	expectedResponse := `{"id":"chatcmpl-123","object":"chat.completion","choices":[{"index":0,"message":{"role":"assistant","content":"Hello!"},"finish_reason":"stop"}]}`

	mockServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/chat/completions" {
			t.Errorf("expected path /chat/completions, got %s", r.URL.Path)
		}
		if r.Method != http.MethodPost {
			t.Errorf("expected POST, got %s", r.Method)
		}
		if r.Header.Get("Content-Type") != "application/json" {
			t.Errorf("expected Content-Type application/json, got %s", r.Header.Get("Content-Type"))
		}
		if r.Header.Get("Authorization") != "Bearer test-api-key" {
			t.Errorf("expected Authorization 'Bearer test-api-key', got '%s'", r.Header.Get("Authorization"))
		}

		var body map[string]interface{}
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Fatalf("failed to decode request body: %v", err)
		}
		if body["model"] != "gpt-4o" {
			t.Errorf("expected model 'gpt-4o', got '%v'", body["model"])
		}
		if body["stream"] != false {
			t.Errorf("expected stream false, got %v", body["stream"])
		}

		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(expectedResponse))
	}))
	defer mockServer.Close()

	provider, err := New(llm.LLMConfig{
		BaseURL:     mockServer.URL,
		APIKey:      "test-api-key",
		DisplayName: "TestProvider",
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	chatReq := llm.ChatRequest{
		Model:    "gpt-4o",
		Messages: []json.RawMessage{json.RawMessage(`{"role":"user","content":"Hi"}`)},
		Stream:   false,
	}

	recorder := httptest.NewRecorder()
	err = provider.Chat(context.Background(), chatReq, recorder)
	if err != nil {
		t.Fatalf("Chat returned error: %v", err)
	}

	if ct := recorder.Header().Get("Content-Type"); ct != "application/json" {
		t.Errorf("expected Content-Type 'application/json', got '%s'", ct)
	}

	respBody := recorder.Body.String()
	if respBody != expectedResponse {
		t.Errorf("expected response body %q, got %q", expectedResponse, respBody)
	}
}

func TestChat_Streaming(t *testing.T) {
	chunk1 := `data: {"id":"chatcmpl-123","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"role":"assistant","content":"Hel"},"finish_reason":null}]}`
	chunk2 := `data: {"id":"chatcmpl-123","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"lo!"},"finish_reason":null}]}`
	chunk3 := `data: {"id":"chatcmpl-123","object":"chat.completion.chunk","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}`
	chunkDone := `data: [DONE]`

	sseBody := fmt.Sprintf("%s\n\n%s\n\n%s\n\n%s\n\n", chunk1, chunk2, chunk3, chunkDone)

	mockServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body map[string]interface{}
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Fatalf("failed to decode request body: %v", err)
		}
		if body["stream"] != true {
			t.Errorf("expected stream true, got %v", body["stream"])
		}

		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("Cache-Control", "no-cache")
		w.WriteHeader(http.StatusOK)
		w.Write([]byte(sseBody))
	}))
	defer mockServer.Close()

	provider, err := New(llm.LLMConfig{
		BaseURL:     mockServer.URL,
		APIKey:      "test-api-key",
		DisplayName: "TestProvider",
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	chatReq := llm.ChatRequest{
		Model:    "gpt-4o",
		Messages: []json.RawMessage{json.RawMessage(`{"role":"user","content":"Hi"}`)},
		Stream:   true,
	}

	recorder := httptest.NewRecorder()
	err = provider.Chat(context.Background(), chatReq, recorder)
	if err != nil {
		t.Fatalf("Chat returned error: %v", err)
	}

	if ct := recorder.Header().Get("Content-Type"); ct != "text/event-stream" {
		t.Errorf("expected Content-Type 'text/event-stream', got '%s'", ct)
	}
	if cc := recorder.Header().Get("Cache-Control"); cc != "no-cache" {
		t.Errorf("expected Cache-Control 'no-cache', got '%s'", cc)
	}

	respBody := recorder.Body.String()
	if !strings.Contains(respBody, "data: {") {
		t.Errorf("expected SSE data in response, got: %s", respBody)
	}
	if !strings.Contains(respBody, "[DONE]") {
		t.Errorf("expected [DONE] in response, got: %s", respBody)
	}
	if !strings.Contains(respBody, "Hel") {
		t.Errorf("expected 'Hel' chunk content in response, got: %s", respBody)
	}
	if !strings.Contains(respBody, "lo!") {
		t.Errorf("expected 'lo!' chunk content in response, got: %s", respBody)
	}
}

func TestChat_UpstreamError(t *testing.T) {
	mockServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusTooManyRequests)
		w.Write([]byte(`{"error":{"message":"rate limit exceeded"}}`))
	}))
	defer mockServer.Close()

	provider, err := New(llm.LLMConfig{
		BaseURL:     mockServer.URL,
		APIKey:      "test-api-key",
		DisplayName: "TestProvider",
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	chatReq := llm.ChatRequest{
		Model:    "gpt-4o",
		Messages: []json.RawMessage{json.RawMessage(`{"role":"user","content":"Hi"}`)},
		Stream:   false,
	}

	recorder := httptest.NewRecorder()
	err = provider.Chat(context.Background(), chatReq, recorder)
	if err == nil {
		t.Fatal("expected error on upstream 429, got nil")
	}
	if !strings.Contains(err.Error(), "429") {
		t.Errorf("expected error to contain '429', got: %s", err.Error())
	}
}

func TestChat_WithTools(t *testing.T) {
	mockServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		bodyBytes, _ := io.ReadAll(r.Body)
		var body map[string]interface{}
		json.Unmarshal(bodyBytes, &body)

		tools, ok := body["tools"]
		if !ok {
			t.Error("expected tools in request body")
		}
		toolsArr, ok := tools.([]interface{})
		if !ok || len(toolsArr) == 0 {
			t.Error("expected non-empty tools array")
		}

		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"id":"chatcmpl-123","choices":[{"message":{"role":"assistant","tool_calls":[{"id":"call_1","type":"function","function":{"name":"get_weather","arguments":"{}"}}]}}]}`))
	}))
	defer mockServer.Close()

	provider, err := New(llm.LLMConfig{
		BaseURL:     mockServer.URL,
		APIKey:      "test-api-key",
		DisplayName: "TestProvider",
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	chatReq := llm.ChatRequest{
		Model:    "gpt-4o",
		Messages: []json.RawMessage{json.RawMessage(`{"role":"user","content":"What's the weather?"}`)},
		Tools:    []json.RawMessage{json.RawMessage(`{"type":"function","function":{"name":"get_weather"}}`)},
		Stream:   false,
	}

	recorder := httptest.NewRecorder()
	err = provider.Chat(context.Background(), chatReq, recorder)
	if err != nil {
		t.Fatalf("Chat returned error: %v", err)
	}

	if !strings.Contains(recorder.Body.String(), "tool_calls") {
		t.Error("expected tool_calls in response")
	}
}

func TestRegister_ListModelsAndChatViaRegistry(t *testing.T) {
	modelsJSON := `{"data":[{"id":"gpt-4o","owned_by":"openai"}]}`
	chatJSON := `{"id":"chatcmpl-1","choices":[{"index":0,"message":{"role":"assistant","content":"hi"},"finish_reason":"stop"}]}`

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case r.Method == http.MethodGet && r.URL.Path == "/models":
			w.Header().Set("Content-Type", "application/json")
			w.Write([]byte(modelsJSON))
		case r.Method == http.MethodPost && r.URL.Path == "/chat/completions":
			w.Header().Set("Content-Type", "application/json")
			w.Write([]byte(chatJSON))
		default:
			t.Errorf("unexpected %s %s", r.Method, r.URL.Path)
			http.NotFound(w, r)
		}
	}))
	defer server.Close()

	for _, providerType := range []string{"openai", "openrouter", "ollama", "custom"} {
		t.Run(providerType, func(t *testing.T) {
			p, err := llm.New(providerType, llm.LLMConfig{
				BaseURL:     server.URL,
				APIKey:      "test-api-key",
				DisplayName: providerType,
			})
			if err != nil {
				t.Fatalf("llm.New(%q): %v", providerType, err)
			}
			if _, ok := p.(*Provider); !ok {
				t.Fatalf("expected *Provider, got %T", p)
			}

			models, err := p.ListModels(context.Background())
			if err != nil {
				t.Fatalf("ListModels: %v", err)
			}
			if len(models) != 1 || models[0].ID != "gpt-4o" {
				t.Fatalf("unexpected models: %+v", models)
			}

			rr := httptest.NewRecorder()
			err = p.Chat(context.Background(), llm.ChatRequest{
				Model:    "gpt-4o",
				Messages: []json.RawMessage{json.RawMessage(`{"role":"user","content":"hi"}`)},
			}, rr)
			if err != nil {
				t.Fatalf("Chat: %v", err)
			}
			if !strings.Contains(rr.Body.String(), `"content":"hi"`) {
				t.Fatalf("chat body missing content, got %s", rr.Body.String())
			}
		})
	}
}
