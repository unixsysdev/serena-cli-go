package orchestrator

import (
	"context"
	"encoding/json"
	"fmt"
	"regexp"
	"strings"

	"github.com/mark3labs/mcp-go/mcp"
	"github.com/sashabaranov/go-openai"
	"github.com/unixsysdev/serena-cli-go/internal/MCP"
	"github.com/unixsysdev/serena-cli-go/internal/config"
	"github.com/unixsysdev/serena-cli-go/internal/llm"
)

// Orchestrator manages the interaction between the LLM and Serena MCP.
type Orchestrator struct {
	config   *config.Config
	llm      *llm.Client
	mcp      *MCP.Client
	messages []openai.ChatCompletionMessage
	tools    []openai.Tool
	events   *EventHandler
	local    map[string]LocalToolHandler
}

// EventHandler allows callers to observe progress and tool usage.
type EventHandler struct {
	OnStatus    func(message string)
	OnToolStart func(name string, args string)
	OnToolEnd   func(name string, result string, isError bool)
}

// LocalToolHandler handles a local tool call without going through MCP.
type LocalToolHandler func(ctx context.Context, arguments map[string]interface{}) (string, error)

// New creates a new orchestrator
func New(cfg *config.Config) (*Orchestrator, error) {
	// Create LLM client.
	llmClient, err := llm.New(&cfg.LLM)
	if err != nil {
		return nil, fmt.Errorf("failed to create LLM client: %w", err)
	}

	// Create MCP client
	mcpClient, err := MCP.New(&cfg.Serena)
	if err != nil {
		return nil, fmt.Errorf("failed to create MCP client: %w", err)
	}

	return &Orchestrator{
		config: cfg,
		llm:    llmClient,
		mcp:    mcpClient,
	}, nil
}

// SetEventHandler sets an optional event handler for progress updates.
func (o *Orchestrator) SetEventHandler(handler *EventHandler) {
	o.events = handler
}

// AddLocalTool registers a local tool and its handler.
func (o *Orchestrator) AddLocalTool(tool openai.Tool, handler LocalToolHandler) {
	if tool.Function == nil || handler == nil {
		return
	}
	if o.local == nil {
		o.local = make(map[string]LocalToolHandler)
	}
	o.local[tool.Function.Name] = handler
	o.tools = append(o.tools, tool)
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
	prompt := `You are Serena CLI, a lean coding assistant powered by Chutes-hosted LLMs and Serena MCP.

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
		Content: wrapUserTask(userMsg),
	})

	o.emitStatus(fmt.Sprintf("thinking (model=%s)", o.llm.Model()))

	if o.config.Debug {
		fmt.Printf("\n=== Sending to LLM ===\nUser: %s\n=====================\n\n", userMsg)
	}

	// Call LLM with tools
	toolChoice := o.selectToolChoice(userMsg)
	content, toolCalls, err := o.llm.ChatWithOptions(ctx, o.llm.Model(), o.messages, o.tools, toolChoice)
	if err != nil {
		return "", fmt.Errorf("LLM chat failed: %w", err)
	}

	content = stripThinkTags(content)

	if o.config.Debug {
		fmt.Printf("\n=== LLM Response ===\nContent: %s\nTool Calls: %d\n====================\n\n", content, len(toolCalls))
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
			o.emitToolStart(toolCall.Function.Name, formatToolArgs(toolCall.Function.Arguments))
			if o.config.Debug {
				fmt.Printf("Calling: %s with args: %s\n", toolCall.Function.Name, toolCall.Function.Arguments)
			}

			result, isError, err := o.executeToolCall(ctx, toolCall)
			if err != nil {
				return "", fmt.Errorf("tool execution failed: %w", err)
			}

			o.emitToolEnd(toolCall.Function.Name, result, isError)

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
			fmt.Printf("=== Calling LLM Again with Tool Results ===\n")
		}

		o.emitStatus(fmt.Sprintf("thinking (model=%s)", o.llm.Model()))

		// Call LLM again with tool results
		content, toolCalls, err = o.llm.ChatWithOptions(ctx, o.llm.Model(), o.messages, o.tools, "auto")
		if err != nil {
			return "", fmt.Errorf("LLM chat with tool results failed: %w", err)
		}

		content = stripThinkTags(content)

		if o.config.Debug {
			fmt.Printf("LLM Response after tools: %s\n", truncateString(content, 200))
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
	return o.llm.Model()
}

// SetModel updates the active model.
func (o *Orchestrator) SetModel(model string) {
	o.config.LLM.Model = model
	o.llm.SetModel(model)
}

// Reset clears the conversation history while keeping the system prompt.
func (o *Orchestrator) Reset() {
	if len(o.messages) > 0 {
		o.messages = o.messages[:1]
	}
}

// SystemPrompt returns the current system prompt.
func (o *Orchestrator) SystemPrompt() string {
	if len(o.messages) == 0 {
		return ""
	}
	return o.messages[0].Content
}

// Messages returns a copy of the current conversation messages.
func (o *Orchestrator) Messages() []openai.ChatCompletionMessage {
	messages := make([]openai.ChatCompletionMessage, len(o.messages))
	copy(messages, o.messages)
	return messages
}

// ReplaceMessages replaces the current conversation messages.
func (o *Orchestrator) ReplaceMessages(messages []openai.ChatCompletionMessage) {
	o.messages = messages
}

// AddContext appends extra context as a system message.
func (o *Orchestrator) AddContext(label string, content string) {
	if strings.TrimSpace(content) == "" {
		return
	}
	msg := fmt.Sprintf("<context source=%q>\n%s\n</context>", label, strings.TrimSpace(content))
	o.messages = append(o.messages, openai.ChatCompletionMessage{
		Role:    openai.ChatMessageRoleSystem,
		Content: msg,
	})
}

// Tools returns the currently loaded tool definitions.
func (o *Orchestrator) Tools() []openai.Tool {
	tools := make([]openai.Tool, len(o.tools))
	copy(tools, o.tools)
	return tools
}

// ConversationStats provides basic context usage estimates.
type ConversationStats struct {
	MessageCount  int
	ToolCallCount int
	CharCount     int
	ApproxTokens  int
}

// ConversationStats returns approximate context usage based on messages and tool calls.
func (o *Orchestrator) ConversationStats() ConversationStats {
	stats := ConversationStats{
		MessageCount: len(o.messages),
	}

	for _, msg := range o.messages {
		stats.CharCount += len(msg.Content)
		if len(msg.ToolCalls) > 0 {
			stats.ToolCallCount += len(msg.ToolCalls)
			for _, call := range msg.ToolCalls {
				stats.CharCount += len(call.Function.Name)
				stats.CharCount += len(call.Function.Arguments)
			}
		}
	}

	if stats.CharCount > 0 {
		stats.ApproxTokens = stats.CharCount / 4
	}

	return stats
}

// Summarize builds a compact summary of the provided text using the compaction model.
func (o *Orchestrator) Summarize(ctx context.Context, text string) (string, error) {
	system := "Summarize the conversation content into a concise, structured summary. " +
		"Preserve key requirements, decisions, file paths, commands, and open questions. " +
		"Use bullets where helpful."
	messages := []openai.ChatCompletionMessage{
		{
			Role:    openai.ChatMessageRoleSystem,
			Content: system,
		},
		{
			Role:    openai.ChatMessageRoleUser,
			Content: text,
		},
	}

	model := o.config.LLM.CompactionModel
	content, _, err := o.llm.ChatWithModel(ctx, model, messages, nil)
	if err != nil {
		return "", err
	}

	return stripThinkTags(content), nil
}

// truncateString truncates a string for display
func truncateString(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen] + "..."
}

func (o *Orchestrator) emitStatus(message string) {
	if o.events != nil && o.events.OnStatus != nil {
		o.events.OnStatus(message)
	}
}

func (o *Orchestrator) emitToolStart(name string, args string) {
	if o.events != nil && o.events.OnToolStart != nil {
		o.events.OnToolStart(name, args)
	}
}

func (o *Orchestrator) emitToolEnd(name string, result string, isError bool) {
	if o.events != nil && o.events.OnToolEnd != nil {
		o.events.OnToolEnd(name, result, isError)
	}
}

func formatToolArgs(raw string) string {
	trimmed := strings.TrimSpace(raw)
	if trimmed == "" {
		return ""
	}
	oneLine := strings.ReplaceAll(trimmed, "\n", " ")
	oneLine = strings.ReplaceAll(oneLine, "\r", " ")
	return truncateString(oneLine, 160)
}

func wrapUserTask(userMsg string) string {
	trimmed := strings.TrimSpace(userMsg)
	if trimmed == "" {
		return "<task></task>"
	}

	return fmt.Sprintf(
		"<task>\n<request>\n%s\n</request>\n<guidance>\n- Focus on the user's request.\n- Use tools only when necessary.\n- Ask a clarifying question if the request is ambiguous.\n</guidance>\n</task>",
		trimmed,
	)
}

func (o *Orchestrator) selectToolChoice(userMsg string) any {
	if len(o.tools) == 0 {
		return nil
	}

	lower := strings.ToLower(userMsg)
	if strings.Contains(lower, "activate project") || strings.Contains(lower, "activate_project") {
		if o.hasTool("activate_project") {
			return openai.ToolChoice{
				Type: openai.ToolTypeFunction,
				Function: openai.ToolFunction{
					Name: "activate_project",
				},
			}
		}
	}

	if strings.Contains(lower, "readme") || strings.Contains(lower, "read file") || strings.Contains(lower, "read the file") {
		if o.hasTool("read_file") {
			return openai.ToolChoice{
				Type: openai.ToolTypeFunction,
				Function: openai.ToolFunction{
					Name: "read_file",
				},
			}
		}
	}

	if strings.Contains(lower, "list files") || strings.Contains(lower, "list dir") || strings.Contains(lower, "list directory") {
		if o.hasTool("list_dir") {
			return openai.ToolChoice{
				Type: openai.ToolTypeFunction,
				Function: openai.ToolFunction{
					Name: "list_dir",
				},
			}
		}
	}

	if strings.Contains(lower, "find file") {
		if o.hasTool("find_file") {
			return openai.ToolChoice{
				Type: openai.ToolTypeFunction,
				Function: openai.ToolFunction{
					Name: "find_file",
				},
			}
		}
	}

	if strings.Contains(lower, "search") {
		if o.hasTool("search_for_pattern") {
			return openai.ToolChoice{
				Type: openai.ToolTypeFunction,
				Function: openai.ToolFunction{
					Name: "search_for_pattern",
				},
			}
		}
	}

	if strings.Contains(lower, "create a script") || strings.Contains(lower, "create script") || strings.Contains(lower, "create a file") || strings.Contains(lower, "write a file") {
		if o.hasTool("create_text_file") {
			return openai.ToolChoice{
				Type: openai.ToolTypeFunction,
				Function: openai.ToolFunction{
					Name: "create_text_file",
				},
			}
		}
	}

	return "auto"
}

func (o *Orchestrator) hasTool(name string) bool {
	for _, tool := range o.tools {
		if tool.Function != nil && tool.Function.Name == name {
			return true
		}
	}
	return false
}

var thinkTagPattern = regexp.MustCompile(`(?s)<think>.*?</think>`)

func stripThinkTags(content string) string {
	if content == "" {
		return content
	}
	clean := thinkTagPattern.ReplaceAllString(content, "")
	return strings.TrimSpace(clean)
}

// executeToolCall executes a single tool call via MCP
func (o *Orchestrator) executeToolCall(ctx context.Context, toolCall openai.ToolCall) (string, bool, error) {
	// Parse tool arguments from JSON
	var args map[string]interface{}
	if toolCall.Function.Arguments != "" {
		if err := json.Unmarshal([]byte(toolCall.Function.Arguments), &args); err != nil {
			return "", false, fmt.Errorf("failed to parse tool arguments: %w", err)
		}
	}

	if handler := o.localToolHandler(toolCall.Function.Name); handler != nil {
		result, err := handler(ctx, args)
		if err != nil {
			return fmt.Sprintf("Error: %s", err.Error()), true, nil
		}
		return result, false, nil
	}

	// Call the tool via MCP
	result, err := o.mcp.CallTool(ctx, toolCall.Function.Name, args)
	if err != nil {
		return "", false, fmt.Errorf("MCP tool call failed: %w", err)
	}

	// Format the result
	return formatToolResult(result), result.IsError, nil
}

func (o *Orchestrator) localToolHandler(name string) LocalToolHandler {
	if o.local == nil {
		return nil
	}
	return o.local[name]
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
