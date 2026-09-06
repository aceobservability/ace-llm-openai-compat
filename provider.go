package openaicompat

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/aceobservability/ace/backend/pkg/llm"
)

// Outbound HTTP uses Go's default http.Client (timeout only). Base URLs are
// checked at save time by Ace's validateBaseURL, not by ssrf.SafeClient or
// ssrf.DatasourceClient — see docs/adr/0003-outbound-http-ssrf-policy-seams.md.
type Provider struct {
	BaseURL     string
	APIKey      string
	DisplayName string
}

func New(cfg llm.LLMConfig) (llm.AIProvider, error) {
	return &Provider{
		BaseURL:     strings.TrimRight(cfg.BaseURL, "/"),
		APIKey:      cfg.APIKey,
		DisplayName: cfg.DisplayName,
	}, nil
}

type openAIModelsResponse struct {
	Data []struct {
		ID      string `json:"id"`
		OwnedBy string `json:"owned_by"`
	} `json:"data"`
}

func (p *Provider) ListModels(ctx context.Context) ([]llm.AIModel, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, p.BaseURL+"/models", nil)
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	if p.APIKey != "" {
		req.Header.Set("Authorization", "Bearer "+p.APIKey)
	}

	client := &http.Client{Timeout: 15 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to reach provider: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("provider returned %d: %s", resp.StatusCode, string(body))
	}

	var raw openAIModelsResponse
	if err := json.NewDecoder(resp.Body).Decode(&raw); err != nil {
		return nil, fmt.Errorf("failed to decode models response: %w", err)
	}

	models := make([]llm.AIModel, 0, len(raw.Data))
	for _, m := range raw.Data {
		models = append(models, llm.AIModel{
			ID:     m.ID,
			Name:   m.ID,
			Vendor: m.OwnedBy,
		})
	}

	return models, nil
}

func (p *Provider) Chat(ctx context.Context, chatReq llm.ChatRequest, w http.ResponseWriter) error {
	body := map[string]interface{}{
		"model":    chatReq.Model,
		"messages": chatReq.Messages,
		"stream":   chatReq.Stream,
	}
	if len(chatReq.Tools) > 0 {
		body["tools"] = chatReq.Tools
	}

	bodyJSON, err := json.Marshal(body)
	if err != nil {
		return fmt.Errorf("failed to marshal request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, p.BaseURL+"/chat/completions", strings.NewReader(string(bodyJSON)))
	if err != nil {
		return fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	if p.APIKey != "" {
		req.Header.Set("Authorization", "Bearer "+p.APIKey)
	}

	client := &http.Client{Timeout: 60 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		return fmt.Errorf("failed to reach provider: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		respBody, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("provider returned %d: %s", resp.StatusCode, string(respBody))
	}

	if !chatReq.Stream {
		w.Header().Set("Content-Type", "application/json")
		io.Copy(w, resp.Body)
		return nil
	}

	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")

	flusher, canFlush := w.(http.Flusher)
	buf := make([]byte, 4096)
	for {
		n, readErr := resp.Body.Read(buf)
		if n > 0 {
			w.Write(buf[:n])
			if canFlush {
				flusher.Flush()
			}
		}
		if readErr != nil {
			break
		}
	}

	return nil
}
