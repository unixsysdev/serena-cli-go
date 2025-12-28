package llm

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"strings"
	"time"

	"github.com/sashabaranov/go-openai"
	"github.com/unixsysdev/serena-cli-go/internal/config"
)

// Client handles LLM API communication.
type Client struct {
	client *openai.Client
	model  string
}

// New creates a new LLM client.
func New(cfg *config.LLMConfig) (*Client, error) {
	if cfg.APIKey == "" {
		return nil, fmt.Errorf("LLM API key is required")
	}

	// Create custom HTTP client with User-Agent header
	httpClient := &http.Client{
		Transport: &userAgentTransport{
			RoundTripper: http.DefaultTransport,
			UserAgent:    "kilo-code/0.1.0",
		},
	}
	if cfg.TimeoutSeconds > 0 {
		httpClient.Timeout = time.Duration(cfg.TimeoutSeconds) * time.Second
	}

	// Create custom config with base URL and custom HTTP client.
	config := openai.DefaultConfig(cfg.APIKey)
	config.BaseURL = cfg.BaseURL
	config.HTTPClient = httpClient

	// Create client with custom config
	client := openai.NewClientWithConfig(config)

	return &Client{
		client: client,
		model:  cfg.Model,
	}, nil
}

// Model returns the current model name.
func (c *Client) Model() string {
	return c.model
}

// SetModel updates the model used for requests.
func (c *Client) SetModel(model string) {
	c.model = model
}

// userAgentTransport wraps an http.RoundTripper to add User-Agent header
type userAgentTransport struct {
	RoundTripper http.RoundTripper
	UserAgent    string
}

func (t *userAgentTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	req.Header.Set("User-Agent", t.UserAgent)
	return t.RoundTripper.RoundTrip(req)
}

// Chat sends a chat completion request
func (c *Client) Chat(ctx context.Context, messages []openai.ChatCompletionMessage, tools []openai.Tool) (string, []openai.ToolCall, error) {
	return c.ChatWithModel(ctx, c.model, messages, tools)
}

// ChatWithModel sends a chat request using an explicit model name.
func (c *Client) ChatWithModel(ctx context.Context, model string, messages []openai.ChatCompletionMessage, tools []openai.Tool) (string, []openai.ToolCall, error) {
	return c.ChatWithOptions(ctx, model, messages, tools, "auto")
}

// ChatWithOptions sends a chat request with explicit tool choice handling.
func (c *Client) ChatWithOptions(ctx context.Context, model string, messages []openai.ChatCompletionMessage, tools []openai.Tool, toolChoice any) (string, []openai.ToolCall, error) {
	if model == "" {
		model = c.model
	}

	req := openai.ChatCompletionRequest{
		Model:       model,
		Messages:    messages,
		Tools:       tools,
		Temperature: 0.7,
	}
	if len(tools) > 0 {
		if toolChoice == nil {
			req.ToolChoice = "auto"
		} else {
			req.ToolChoice = toolChoice
		}
	}

	resp, err := c.client.CreateChatCompletion(ctx, req)
	if err != nil {
		return "", nil, fmt.Errorf("chat completion failed for model %q: %s", model, formatLLMError(err))
	}

	if len(resp.Choices) == 0 {
		return "", nil, fmt.Errorf("no response from LLM")
	}

	content := resp.Choices[0].Message.Content
	toolCalls := resp.Choices[0].Message.ToolCalls

	return content, toolCalls, nil
}

func formatLLMError(err error) string {
	if err == nil {
		return "unknown error"
	}

	var apiErr *openai.APIError
	if errors.As(err, &apiErr) {
		return formatAPIError(apiErr)
	}

	var reqErr *openai.RequestError
	if errors.As(err, &reqErr) {
		base := fmt.Sprintf("status %d", reqErr.HTTPStatusCode)
		if reqErr.Err != nil {
			return base + ": " + formatLLMError(reqErr.Err)
		}
		return base + ": empty error response from provider"
	}

	return err.Error()
}

func formatAPIError(apiErr *openai.APIError) string {
	if apiErr == nil {
		return "unknown API error"
	}
	parts := make([]string, 0, 4)
	if apiErr.Message != "" {
		parts = append(parts, apiErr.Message)
	}
	if apiErr.Type != "" {
		parts = append(parts, "type="+apiErr.Type)
	}
	if apiErr.Param != nil && *apiErr.Param != "" {
		parts = append(parts, "param="+*apiErr.Param)
	}
	if apiErr.Code != nil {
		parts = append(parts, fmt.Sprintf("code=%v", apiErr.Code))
	}
	if apiErr.HTTPStatusCode > 0 {
		parts = append([]string{fmt.Sprintf("status %d", apiErr.HTTPStatusCode)}, parts...)
	}
	if len(parts) == 0 {
		return "unknown API error"
	}
	return strings.Join(parts, ", ")
}
