package MCP

import (
	"context"
	"fmt"

	"github.com/mark3labs/mcp-go/client"
	"github.com/mark3labs/mcp-go/client/transport"
	"github.com/mark3labs/mcp-go/mcp"
	"github.com/unixsysdev/serena-cli-go/internal/config"
)

// Client handles MCP communication with Serena
type Client struct {
	client       *client.Client
	Instructions string // Server instructions from initialization
}

// New creates a new MCP client
func New(cfg *config.SerenaConfig) (*Client, error) {
	// Build command args
	args := cfg.Args
	if len(args) == 0 {
		args = []string{
			"--from", "git+https://github.com/oraios/serena",
			"serena", "start-mcp-server",
		}
	}

	// Add context flag
	args = append(args, "--context", cfg.Context)

	// Add project path if specified
	if cfg.ProjectPath != "" {
		args = append(args, "--project", cfg.ProjectPath)
	}

	// Create stdio transport
	stdio := transport.NewStdio(cfg.Command, nil, args...)

	// Create MCP client
	mcpClient := client.NewClient(stdio)

	return &Client{
		client: mcpClient,
	}, nil
}

// Connect starts the MCP server and initializes the session
func (c *Client) Connect() error {
	ctx := context.Background()

	// Start the transport
	if err := c.client.Start(ctx); err != nil {
		return fmt.Errorf("failed to start MCP client: %w", err)
	}

	// Initialize the MCP session
	initRequest := mcp.InitializeRequest{
		Params: mcp.InitializeParams{
			ProtocolVersion: mcp.LATEST_PROTOCOL_VERSION,
			ClientInfo: mcp.Implementation{
				Name:    "serena-cli",
				Version: "1.0.0",
			},
			Capabilities: mcp.ClientCapabilities{},
		},
	}

	result, err := c.client.Initialize(ctx, initRequest)
	if err != nil {
		return fmt.Errorf("MCP initialization failed: %w", err)
	}

	// Store server instructions for later use
	c.Instructions = result.Instructions

	return nil
}

// Close closes the connection to the MCP server
func (c *Client) Close() error {
	return c.client.Close()
}

// ListTools lists all available tools from the MCP server
func (c *Client) ListTools(ctx context.Context) ([]mcp.Tool, error) {
	request := mcp.ListToolsRequest{}
	result, err := c.client.ListTools(ctx, request)
	if err != nil {
		return nil, fmt.Errorf("failed to list tools: %w", err)
	}

	return result.Tools, nil
}

// CallTool calls a specific tool on the MCP server
func (c *Client) CallTool(ctx context.Context, name string, arguments map[string]interface{}) (*mcp.CallToolResult, error) {
	request := mcp.CallToolRequest{
		Params: mcp.CallToolParams{
			Name:      name,
			Arguments: arguments,
		},
	}

	result, err := c.client.CallTool(ctx, request)
	if err != nil {
		return nil, fmt.Errorf("failed to call tool %s: %w", name, err)
	}

	return result, nil
}

// GetClient returns the underlying MCP client for advanced usage
func (c *Client) GetClient() *client.Client {
	return c.client
}
