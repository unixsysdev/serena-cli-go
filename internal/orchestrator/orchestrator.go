package orchestrator

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/mark3labs/mcp-go/mcp"
	"github.com/sashabaranov/go-openai"
	"github.com/unixsysdev/serena-cli-go/internal/GLM"
	"github.com/unixsysdev/serena-cli-go/internal/MCP"
	"github.com/unixsysdev/serena-cli-go/internal/config"
)

// Orchestrator manages the interaction between GLM and Serena MCP
type Orchestrator struct {
	config   *config.Config
	glm      *GLM.Client
	mcp      *MCP.Client
	messages []openai.ChatCompletionMessage
	tools    []openai.Tool
}

// New creates a new orchestrator
func New(cfg *config.Config) (*Orchestrator, error) {
	// Create GLM client
	glm, err := GLM.New(&cfg.GLM)
	if err != nil {
		return nil, fmt.Errorf("failed to create GLM client: %w", err)
	}

	// Create MCP client
	mcpClient, err := MCP.New(&cfg.Serena)
	if err != nil {
		return nil, fmt.Errorf("failed to create MCP client: %w", err)
	}

	return &Orchestrator{
		config: cfg,
		glm:    glm,
		mcp:    mcpClient,
	}, nil
}

// Initialize sets up connections and loads available tools
func (o *Orchestrator) Initialize() error {
	fmt.Print("Connecting to Serena MCP... ")

	if err := o.mcp.Connect(); err != nil {
		return fmt.Errorf("failed to connect: %w", err)
	}

	fmt.Println("✓")

	// Load tools from MCP
	fmt.Print("Loading tools from Serena... ")
	ctx := context.Background()
	mcpTools, err := o.mcp.ListTools(ctx)
	if err != nil {
		return fmt.Errorf("failed to list tools: %w", err)
	}

	// Convert MCP tools to OpenAI format
	o.tools = convertMCPToolsToOpenAI(mcpTools)
	fmt.Printf("✓ (%d tools loaded)\n", len(o.tools))

	// Use Serena's instructions as the system prompt
	systemPrompt := o.mcp.Instructions
	if systemPrompt == "" {
		// Fallback if no instructions provided
		systemPrompt = o.buildFallbackPrompt(mcpTools)
	}

	// Debug: Print what Serena sent us
	if o.config.Debug {
		fmt.Printf("\n=== Serena's Instructions ===\n%s\n============================\n\n", systemPrompt)
	}

	o.messages = []openai.ChatCompletionMessage{
		{
			Role:    openai.ChatMessageRoleSystem,
			Content: systemPrompt,
		},
	}

	return nil
}

// buildFallbackPrompt creates a fallback system prompt when Serena doesn't provide one
func (o *Orchestrator) buildFallbackPrompt(tools []mcp.Tool) string {
	prompt := `You are Serena CLI, a lean coding assistant powered by GLM 4.7 and Serena MCP.

You have access to powerful tools through the Serena MCP server. These tools can help you:
- Read and analyze code
- Edit files with symbolic precision
- Search codebases intelligently
- Understand code structure and relationships
- Run tests and builds
- And much more

Available tools:
`
	for _, tool := range tools {
		prompt += fmt.Sprintf("- %s: %s\n", tool.Name, tool.Description)
	}

	prompt += `
When the user asks you to do something:
1. Think about what tools you need
2. Use the available tools appropriately
3. Interpret the results
4. Provide a helpful response

Be direct and efficient. Focus on getting things done.`
	return prompt
}

// Chat processes a user message and returns the response
func (o *Orchestrator) Chat(ctx context.Context, userMsg string) (string, error) {
	// Add user message
	o.messages = append(o.messages, openai.ChatCompletionMessage{
		Role:    openai.ChatMessageRoleUser,
		Content: userMsg,
	})

	if o.config.Debug {
		fmt.Printf("\n=== Sending to GLM ===\nUser: %s\n=====================\n\n", userMsg)
	}

	// Call GLM with tools
	content, toolCalls, err := o.glm.Chat(ctx, o.messages, o.tools)
	if err != nil {
		return "", fmt.Errorf("GLM chat failed: %w", err)
	}

	if o.config.Debug {
		fmt.Printf("\n=== GLM Response ===\nContent: %s\nTool Calls: %d\n====================\n\n", content, len(toolCalls))
		for i, tc := range toolCalls {
			fmt.Printf("  Tool %d: %s\n", i+1, tc.Function.Name)
		}
	}

	// Add assistant response
	o.messages = append(o.messages, openai.ChatCompletionMessage{
		Role:      openai.ChatMessageRoleAssistant,
		Content:   content,
		ToolCalls: toolCalls,
	})

	// If there are tool calls, execute them
	for len(toolCalls) > 0 {
		if o.config.Debug {
			fmt.Printf("\n=== Executing %d Tool Calls ===\n", len(toolCalls))
		}

		// Execute each tool call
		for _, toolCall := range toolCalls {
			if o.config.Debug {
				fmt.Printf("Calling: %s with args: %s\n", toolCall.Function.Name, toolCall.Function.Arguments)
			}

			result, err := o.executeToolCall(ctx, toolCall)
			if err != nil {
				return "", fmt.Errorf("tool execution failed: %w", err)
			}

			if o.config.Debug {
				fmt.Printf("Result: %s\n", truncateString(result, 200))
			}

			// Add tool result message
			o.messages = append(o.messages, openai.ChatCompletionMessage{
				Role:       openai.ChatMessageRoleTool,
				Content:    result,
				ToolCallID: toolCall.ID,
			})
		}

		if o.config.Debug {
			fmt.Printf("=== Calling GLM Again with Tool Results ===\n")
		}

		// Call GLM again with tool results
		content, toolCalls, err = o.glm.Chat(ctx, o.messages, o.tools)
		if err != nil {
			return "", fmt.Errorf("GLM chat with tool results failed: %w", err)
		}

		if o.config.Debug {
			fmt.Printf("GLM Response after tools: %s\n", truncateString(content, 200))
		}

		// Add assistant response
		o.messages = append(o.messages, openai.ChatCompletionMessage{
			Role:      openai.ChatMessageRoleAssistant,
			Content:   content,
			ToolCalls: toolCalls,
		})
	}

	return content, nil
}

// Model returns the active model name.
func (o *Orchestrator) Model() string {
	return o.glm.Model()
}

// SetModel updates the active model.
func (o *Orchestrator) SetModel(model string) {
	o.config.GLM.Model = model
	o.glm.SetModel(model)
}

// Reset clears the conversation history while keeping the system prompt.
func (o *Orchestrator) Reset() {
	if len(o.messages) > 0 {
		o.messages = o.messages[:1]
	}
}

// truncateString truncates a string for display
func truncateString(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen] + "..."
}

// executeToolCall executes a single tool call via MCP
func (o *Orchestrator) executeToolCall(ctx context.Context, toolCall openai.ToolCall) (string, error) {
	// Parse tool arguments from JSON
	var args map[string]interface{}
	if toolCall.Function.Arguments != "" {
		if err := json.Unmarshal([]byte(toolCall.Function.Arguments), &args); err != nil {
			return "", fmt.Errorf("failed to parse tool arguments: %w", err)
		}
	}

	// Call the tool via MCP
	result, err := o.mcp.CallTool(ctx, toolCall.Function.Name, args)
	if err != nil {
		return "", fmt.Errorf("MCP tool call failed: %w", err)
	}

	// Format the result
	return formatToolResult(result), nil
}

// Close cleans up connections
func (o *Orchestrator) Close() error {
	return o.mcp.Close()
}

// convertMCPToolsToOpenAI converts MCP tools to OpenAI format
func convertMCPToolsToOpenAI(mcpTools []mcp.Tool) []openai.Tool {
	tools := make([]openai.Tool, 0, len(mcpTools))

	for _, mcpTool := range mcpTools {
		functionDef := openai.FunctionDefinition{
			Name:        mcpTool.Name,
			Description: mcpTool.Description,
		}

		// Convert input schema if present
		// InputSchema is a ToolArgumentsSchema with Type, Properties, Required fields
		// We need to convert it to a map for OpenAI
		if mcpTool.InputSchema.Type != "" || len(mcpTool.InputSchema.Properties) > 0 {
			schemaMap := map[string]interface{}{
				"type": mcpTool.InputSchema.Type,
			}
			if len(mcpTool.InputSchema.Properties) > 0 {
				schemaMap["properties"] = mcpTool.InputSchema.Properties
			}
			if len(mcpTool.InputSchema.Required) > 0 {
				schemaMap["required"] = mcpTool.InputSchema.Required
			}
			functionDef.Parameters = schemaMap
		} else if mcpTool.RawInputSchema != nil {
			// Use raw schema if available
			functionDef.Parameters = mcpTool.RawInputSchema
		}

		tool := openai.Tool{
			Type:     openai.ToolTypeFunction,
			Function: &functionDef,
		}

		tools = append(tools, tool)
	}

	return tools
}

// formatToolResult formats a tool result as a string
func formatToolResult(result *mcp.CallToolResult) string {
	if result == nil {
		return ""
	}

	var output string
	for _, content := range result.Content {
		switch c := content.(type) {
		case mcp.TextContent:
			output += c.Text
		case mcp.ImageContent:
			output += fmt.Sprintf("[Image: %s]", c.MIMEType)
		case mcp.EmbeddedResource:
			if text, ok := c.Resource.(mcp.TextResourceContents); ok {
				output += text.Text
			}
		}
	}

	// Check for errors
	if result.IsError {
		return fmt.Sprintf("Error: %s", output)
	}

	return output
}
