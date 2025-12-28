package GLM

import (
	"context"
	"fmt"
	"net/http"

	"github.com/sashabaranov/go-openai"
	"github.com/unixsysdev/serena-cli-go/internal/config"
)

// Client handles GLM API communication
type Client struct {
	client *openai.Client
	model  string
}

// New creates a new GLM client
func New(cfg *config.GLMConfig) (*Client, error) {
	if cfg.APIKey == "" {
		return nil, fmt.Errorf("GLM API key is required")
	}

	// Create custom HTTP client with User-Agent header
	httpClient := &http.Client{
		Transport: &userAgentTransport{
			RoundTripper: http.DefaultTransport,
			UserAgent:    "kilo-code/0.1.0",
		},
	}

	// Create custom config with base URL and custom HTTP client
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
	// Create request
	req := openai.ChatCompletionRequest{
		Model:       c.model,
		Messages:    messages,
		Tools:       tools,
		Temperature: 0.7,
	}

	// Send request
	resp, err := c.client.CreateChatCompletion(ctx, req)
	if err != nil {
		return "", nil, fmt.Errorf("chat completion failed: %w", err)
	}

	if len(resp.Choices) == 0 {
		return "", nil, fmt.Errorf("no response from GLM")
	}

	// Extract content and tool calls
	content := resp.Choices[0].Message.Content
	toolCalls := resp.Choices[0].Message.ToolCalls

	return content, toolCalls, nil
}
