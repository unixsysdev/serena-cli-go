package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/peterh/liner"
	"github.com/unixsysdev/serena-cli-go/internal/config"
	"github.com/unixsysdev/serena-cli-go/internal/orchestrator"
)

var version = "dev"

const (
	contextLimitTokens = 200000
	maxToolHistory     = 25
	maxToolPreview     = 200
	maxToolStore       = 2000
	maxContextFileSize = 200000
)

const autoCompactThreshold = 0.9

const (
	colorReset  = "\x1b[0m"
	colorBold   = "\x1b[1m"
	colorRed    = "\x1b[31m"
	colorGreen  = "\x1b[32m"
	colorYellow = "\x1b[33m"
	colorBlue   = "\x1b[34m"
	colorCyan   = "\x1b[36m"
	colorGray   = "\x1b[90m"
)

var availableModels = []string{
	"deepseek-ai/DeepSeek-V3.2-Speciale-TEE",
	"MiniMaxAI/MiniMax-M2.1-TEE",
	"Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8-TEE",
	"moonshotai/Kimi-K2-Thinking-TEE",
	"moonshotai/Kimi-K2-Instruct-0905",
	"deepseek-ai/DeepSeek-V3.2-TEE",
	"zai-org/GLM-4.7-TEE",
}

func main() {
	var showConfig bool
	var showVersion bool

	flag.BoolVar(&showConfig, "config", false, "Print resolved configuration and exit")
	flag.BoolVar(&showVersion, "version", false, "Print version and exit")
	flag.Parse()

	if showVersion {
		fmt.Println(version)
		return
	}

	cfg, err := config.LoadWithOptions(config.LoadOptions{SkipValidation: showConfig})
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}

	if showConfig {
		if err := printConfig(cfg); err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		return
	}

	orch, err := orchestrator.New(cfg)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}

	if err := orch.Initialize(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	defer func() {
		_ = orch.Close()
	}()

	sessions, err := initSessionState(cfg, orch)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}

	ui := attachConsoleUI(orch)

	ctx := context.Background()

	if flag.NArg() > 0 {
		prompt := strings.Join(flag.Args(), " ")
		resp, err := orch.Chat(ctx, prompt)
		ui.StopSpinner()
		_ = sessions.SaveFromOrch(orch)
		if err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		fmt.Println(resp)
		return
	}

	if err := runREPL(ctx, orch, cfg, ui, sessions); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func runREPL(ctx context.Context, orch *orchestrator.Orchestrator, cfg *config.Config, ui *ConsoleUI, sessions *SessionState) error {
	line := liner.NewLiner()
	line.SetCtrlCAborts(true)
	defer func() {
		_ = line.Close()
	}()

	fmt.Fprint(os.Stderr, formatBanner("Model", orch.Model(), "(use /model to switch)"))
	fmt.Fprint(os.Stderr, formatBanner("Session", sessions.Current(), "(use /session to manage)"))
	for {
		prompt := promptString(cfg, orch, sessions)
		input, err := line.Prompt(prompt)
		if err != nil {
			if err == liner.ErrPromptAborted {
				fmt.Println()
				continue
			}
			if err == io.EOF {
				return nil
			}
			return err
		}

		text := strings.TrimSpace(input)
		if text == "" {
			continue
		}
		line.AppendHistory(text)

		if strings.HasPrefix(text, "@context") {
			if err := handleContextImport(text, orch, sessions); err != nil {
				fmt.Println(err)
			} else {
				fmt.Println("Context file added.")
			}
			continue
		}
		if strings.HasPrefix(text, "/") {
			exit, err := handleCommand(ctx, text, orch, cfg, ui, sessions)
			if err != nil {
				fmt.Println(err)
			}
			if exit {
				return nil
			}
			continue
		}
		if text == "exit" || text == "quit" {
			return nil
		}

		resp, err := orch.Chat(ctx, text)
		ui.StopSpinner()
		if err := maybeAutoCompact(ctx, orch, sessions); err != nil {
			fmt.Fprintln(os.Stderr, err)
		}
		_ = sessions.SaveFromOrch(orch)
		if err != nil {
			return err
		}

		fmt.Println(resp)
	}
}

func handleCommand(ctx context.Context, line string, orch *orchestrator.Orchestrator, cfg *config.Config, ui *ConsoleUI, sessions *SessionState) (bool, error) {
	fields := strings.Fields(line)
	if len(fields) == 0 {
		return false, nil
	}

	cmd := strings.TrimPrefix(fields[0], "/")
	args := fields[1:]

	switch cmd {
	case "exit", "quit":
		return true, nil
	case "help":
		printHelp()
		return false, nil
	case "model", "models":
		return false, handleModelCommand(cmd, args, orch, sessions)
	case "tools":
		return false, listTools(orch)
	case "status":
		return false, printStatus(orch, cfg, sessions)
	case "context":
		return false, printContext(orch)
	case "trace":
		return false, ui.PrintTrace(args)
	case "session":
		return false, handleSessionCommand(args, orch, sessions, ui)
	case "compact":
		return false, compactSession(ctx, orch, sessions)
	case "toolmode":
		return false, handleToolModeCommand(args, orch)
	case "clear":
		clearScreen()
		return false, nil
	case "config":
		return false, printConfig(cfg)
	case "reset":
		orch.Reset()
		_ = sessions.SaveFromOrch(orch)
		fmt.Println("Conversation reset.")
		return false, nil
	default:
		return false, fmt.Errorf("unknown command: %s (try /help)", fields[0])
	}
}

func handleModelCommand(cmd string, args []string, orch *orchestrator.Orchestrator, sessions *SessionState) error {
	if cmd == "models" || len(args) == 0 {
		listModels(orch.Model())
		return nil
	}

	arg := strings.TrimSpace(strings.Join(args, " "))
	if arg == "" || strings.EqualFold(arg, "list") {
		listModels(orch.Model())
		return nil
	}

	if idx, err := strconv.Atoi(arg); err == nil {
		if idx < 1 || idx > len(availableModels) {
			return fmt.Errorf("model index out of range: %d", idx)
		}
		model := availableModels[idx-1]
		orch.SetModel(model)
		_ = sessions.SaveFromOrch(orch)
		fmt.Printf("Model set to %s\n", model)
		return nil
	}

	for _, model := range availableModels {
		if strings.EqualFold(model, arg) {
			orch.SetModel(model)
			_ = sessions.SaveFromOrch(orch)
			fmt.Printf("Model set to %s\n", model)
			return nil
		}
	}

	return fmt.Errorf("unknown model: %s (try /model to list)", arg)
}

func listModels(current string) {
	fmt.Println("Available models:")
	for i, model := range availableModels {
		marker := " "
		if model == current {
			marker = "*"
		}
		fmt.Printf("%s %d) %s\n", marker, i+1, model)
	}
	fmt.Printf("Current: %s\n", current)
	fmt.Println("Use /model <number|name> to switch.")
}

func printHelp() {
	fmt.Println("Commands:")
	fmt.Println("  /help           Show this help")
	fmt.Println("  /model          List models")
	fmt.Println("  /model <value>  Switch model by index or name")
	fmt.Println("  /models         Alias for /model")
	fmt.Println("  /tools          List available tools")
	fmt.Println("  /status         Show current status")
	fmt.Println("  /context        Show context usage")
	fmt.Println("  /trace [n]      Show recent tool calls")
	fmt.Println("  /session ...    Manage sessions (list/new/switch/delete)")
	fmt.Println("  /compact        Compact older context into a summary")
	fmt.Println("  /toolmode       Show or set tool selection mode")
	fmt.Println("  /clear          Clear the screen")
	fmt.Println("  /config         Show resolved config (API key masked)")
	fmt.Println("  /reset          Clear the conversation context")
	fmt.Println("  /exit, /quit    Exit the CLI")
	fmt.Println("  @context <file> Append a file to the system context")
}

func attachConsoleUI(orch *orchestrator.Orchestrator) *ConsoleUI {
	ui := NewConsoleUI(os.Stderr)
	orch.SetEventHandler(ui.Handler())
	return ui
}

func listTools(orch *orchestrator.Orchestrator) error {
	tools := orch.Tools()
	if len(tools) == 0 {
		fmt.Println("No tools loaded.")
		return nil
	}
	fmt.Println("Available tools:")
	for _, tool := range tools {
		if tool.Function == nil {
			continue
		}
		fmt.Printf("- %s: %s\n", tool.Function.Name, tool.Function.Description)
	}
	return nil
}

func printContext(orch *orchestrator.Orchestrator) error {
	stats := orch.ConversationStats()
	percent := (float64(stats.ApproxTokens) / float64(contextLimitTokens)) * 100
	fmt.Printf("Messages: %d\n", stats.MessageCount)
	fmt.Printf("Tool calls: %d\n", stats.ToolCallCount)
	fmt.Printf("Characters: %d\n", stats.CharCount)
	fmt.Printf("Approx tokens: %d / %d (%.1f%%)\n", stats.ApproxTokens, contextLimitTokens, percent)
	if percent >= 85 {
		fmt.Println("Warning: context usage is high; consider /reset.")
	}
	return nil
}

func maybeAutoCompact(ctx context.Context, orch *orchestrator.Orchestrator, sessions *SessionState) error {
	stats := orch.ConversationStats()
	if stats.ApproxTokens < int(float64(contextLimitTokens)*autoCompactThreshold) {
		return nil
	}
	fmt.Fprintln(os.Stderr, "Context is large; auto-compacting...")
	return compactSession(ctx, orch, sessions)
}

func printStatus(orch *orchestrator.Orchestrator, cfg *config.Config, sessions *SessionState) error {
	stats := orch.ConversationStats()
	percent := (float64(stats.ApproxTokens) / float64(contextLimitTokens)) * 100
	fmt.Printf("Model: %s\n", orch.Model())
	if cfg.LLM.CompactionModel != "" && cfg.LLM.CompactionModel != orch.Model() {
		fmt.Printf("Compaction model: %s\n", cfg.LLM.CompactionModel)
	}
	fmt.Printf("Project: %s\n", cfg.Serena.ProjectPath)
	contextLabel := cfg.Serena.Context
	if strings.TrimSpace(contextLabel) == "" {
		contextLabel = "(auto)"
	}
	fmt.Printf("Context: %s\n", contextLabel)
	fmt.Printf("Tool mode: %s\n", orch.ToolMode())
	fmt.Printf("Tools loaded: %d\n", len(orch.Tools()))
	fmt.Printf("Session: %s\n", sessions.Current())
	fmt.Printf("Approx tokens: %d / %d (%.1f%%)\n", stats.ApproxTokens, contextLimitTokens, percent)
	return nil
}

func truncateLine(line string, maxLen int) string {
	trimmed := strings.TrimSpace(line)
	if len(trimmed) <= maxLen {
		return trimmed
	}
	return trimmed[:maxLen] + "..."
}

func printConfig(cfg *config.Config) error {
	display := map[string]interface{}{
		"llm": map[string]string{
			"api_key":          maskKey(cfg.LLM.APIKey),
			"base_url":         cfg.LLM.BaseURL,
			"model":            cfg.LLM.Model,
			"compaction_model": cfg.LLM.CompactionModel,
		},
		"serena": map[string]interface{}{
			"project_path": cfg.Serena.ProjectPath,
			"context":      cfg.Serena.Context,
			"command":      cfg.Serena.Command,
			"args":         cfg.Serena.Args,
		},
		"debug": cfg.Debug,
	}

	if len(cfg.Serena.Env) > 0 {
		serena := display["serena"].(map[string]interface{})
		serena["env"] = cfg.Serena.Env
	}

	data, err := json.MarshalIndent(display, "", "  ")
	if err != nil {
		return err
	}

	fmt.Println(string(data))
	return nil
}

func maskKey(key string) string {
	if key == "" {
		return ""
	}
	if len(key) <= 8 {
		return "********"
	}
	return key[:4] + "..." + key[len(key)-4:]
}

type ToolEvent struct {
	Name       string
	Args       string
	Result     string
	ResultSize int
	IsError    bool
	Started    time.Time
	Duration   time.Duration
}

type ConsoleUI struct {
	out         *os.File
	color       bool
	mu          sync.Mutex
	spinnerStop chan struct{}
	toolHistory []ToolEvent
	currentTool *ToolEvent
}

func NewConsoleUI(out *os.File) *ConsoleUI {
	return &ConsoleUI{
		out:   out,
		color: useColor(),
	}
}

func (ui *ConsoleUI) Handler() *orchestrator.EventHandler {
	return &orchestrator.EventHandler{
		OnStatus:    ui.handleStatus,
		OnToolStart: ui.handleToolStart,
		OnToolEnd:   ui.handleToolEnd,
	}
}

func (ui *ConsoleUI) StopSpinner() {
	ui.mu.Lock()
	defer ui.mu.Unlock()
	ui.stopSpinnerLocked()
}

func (ui *ConsoleUI) handleStatus(message string) {
	ui.mu.Lock()
	defer ui.mu.Unlock()

	if isThinkingStatus(message) {
		ui.startSpinnerLocked(normalizeStatus(message))
		return
	}

	ui.stopSpinnerLocked()
	fmt.Fprintf(ui.out, "%s %s\n", ui.colorize(colorCyan, "[status]"), message)
}

func (ui *ConsoleUI) handleToolStart(name string, args string) {
	ui.mu.Lock()
	defer ui.mu.Unlock()

	ui.stopSpinnerLocked()

	event := &ToolEvent{
		Name:    name,
		Args:    args,
		Started: time.Now(),
	}
	ui.currentTool = event

	if args == "" {
		fmt.Fprintf(ui.out, "%s %s\n", ui.colorize(colorBlue, "[tool]"), name)
		return
	}
	fmt.Fprintf(ui.out, "%s %s %s\n", ui.colorize(colorBlue, "[tool]"), name, ui.colorize(colorGray, args))
}

func (ui *ConsoleUI) handleToolEnd(name string, result string, isError bool) {
	ui.mu.Lock()
	defer ui.mu.Unlock()

	ui.stopSpinnerLocked()

	event := ui.currentTool
	if event == nil {
		event = &ToolEvent{Name: name}
	}

	event.ResultSize = len(result)
	event.Result = truncateText(result, maxToolStore)
	event.IsError = isError
	if !event.Started.IsZero() {
		event.Duration = time.Since(event.Started)
	}

	ui.currentTool = nil
	ui.appendToolEvent(*event)

	duration := formatDuration(event.Duration)
	if isError {
		fmt.Fprintf(ui.out, "%s %s %s (%s): %s\n", ui.colorize(colorBlue, "[tool]"), name, ui.colorize(colorRed, "error"), duration, truncateLine(singleLine(result), maxToolPreview))
		return
	}
	if event.ResultSize > 0 {
		fmt.Fprintf(ui.out, "%s %s %s (%s, %d chars)\n", ui.colorize(colorBlue, "[tool]"), name, ui.colorize(colorGreen, "done"), duration, event.ResultSize)
		return
	}
	fmt.Fprintf(ui.out, "%s %s %s (%s)\n", ui.colorize(colorBlue, "[tool]"), name, ui.colorize(colorGreen, "done"), duration)
}

func (ui *ConsoleUI) PrintTrace(args []string) error {
	limit := 5
	if len(args) > 0 {
		arg := strings.TrimSpace(args[0])
		if strings.EqualFold(arg, "all") {
			limit = maxToolHistory
		} else if value, err := strconv.Atoi(arg); err == nil && value > 0 {
			limit = value
		} else {
			return fmt.Errorf("usage: /trace [n|all]")
		}
	}

	history := ui.snapshotHistory()
	if len(history) == 0 {
		fmt.Println("No tool calls recorded yet.")
		return nil
	}

	if limit > len(history) {
		limit = len(history)
	}

	fmt.Printf("Last %d tool calls:\n", limit)
	start := len(history) - limit
	for i := start; i < len(history); i++ {
		event := history[i]
		status := "ok"
		if event.IsError {
			status = "error"
		}
		detail := ""
		if event.Args != "" {
			detail = fmt.Sprintf(" args=%s", event.Args)
		}
		preview := ""
		if event.Result != "" {
			preview = fmt.Sprintf(" result=%s", truncateLine(singleLine(event.Result), maxToolPreview))
		}
		fmt.Printf("- %s %s (%s)%s%s\n", event.Name, status, formatDuration(event.Duration), detail, preview)
	}

	return nil
}

func (ui *ConsoleUI) snapshotHistory() []ToolEvent {
	ui.mu.Lock()
	defer ui.mu.Unlock()

	history := make([]ToolEvent, len(ui.toolHistory))
	copy(history, ui.toolHistory)
	return history
}

func (ui *ConsoleUI) appendToolEvent(event ToolEvent) {
	ui.toolHistory = append(ui.toolHistory, event)
	if len(ui.toolHistory) > maxToolHistory {
		ui.toolHistory = ui.toolHistory[len(ui.toolHistory)-maxToolHistory:]
	}
}

func (ui *ConsoleUI) startSpinnerLocked(message string) {
	ui.stopSpinnerLocked()

	if message == "" {
		message = "..."
	}
	stop := make(chan struct{})
	ui.spinnerStop = stop

	go func(msg string, stopCh chan struct{}) {
		ticker := time.NewTicker(400 * time.Millisecond)
		defer ticker.Stop()

		frames := []string{"", ".", "..", "..."}
		frame := 0
		for {
			select {
			case <-stopCh:
				fmt.Fprint(ui.out, "\r")
				fmt.Fprint(ui.out, strings.Repeat(" ", len(msg)+20))
				fmt.Fprint(ui.out, "\r")
				return
			case <-ticker.C:
				fmt.Fprintf(ui.out, "\r%s %s%s", ui.colorize(colorYellow, "[thinking]"), msg, frames[frame%len(frames)])
				frame++
			}
		}
	}(message, stop)
}

func (ui *ConsoleUI) stopSpinnerLocked() {
	if ui.spinnerStop == nil {
		return
	}
	close(ui.spinnerStop)
	ui.spinnerStop = nil
}

func isThinkingStatus(message string) bool {
	return strings.HasPrefix(strings.ToLower(message), "thinking")
}

func normalizeStatus(message string) string {
	trimmed := strings.TrimSpace(message)
	if !isThinkingStatus(trimmed) {
		return trimmed
	}

	trimmed = strings.TrimSpace(strings.TrimPrefix(trimmed, "thinking"))
	trimmed = strings.Trim(trimmed, "()")
	return trimmed
}

func (ui *ConsoleUI) colorize(code string, text string) string {
	if !ui.color {
		return text
	}
	return code + text + colorReset
}

func useColor() bool {
	if os.Getenv("NO_COLOR") != "" || os.Getenv("SERENA_NO_COLOR") != "" {
		return false
	}
	return true
}

func truncateText(text string, maxLen int) string {
	if maxLen <= 0 || len(text) <= maxLen {
		return text
	}
	return text[:maxLen] + "..."
}

func singleLine(text string) string {
	fields := strings.Fields(text)
	if len(fields) == 0 {
		return ""
	}
	return strings.Join(fields, " ")
}

func formatDuration(d time.Duration) string {
	if d <= 0 {
		return "0s"
	}
	if d < time.Second {
		return d.Round(time.Millisecond).String()
	}
	return d.Round(10 * time.Millisecond).String()
}

func clearScreen() {
	fmt.Print("\033[2J\033[H")
}

func promptString(cfg *config.Config, orch *orchestrator.Orchestrator, sessions *SessionState) string {
	project := cfg.Serena.ProjectPath
	if strings.TrimSpace(project) == "" || project == "." {
		if cwd, err := os.Getwd(); err == nil {
			project = cwd
		}
	}
	if project != "" {
		project = filepath.Base(project)
	}
	if project == "" {
		project = "serena"
	}

	model := shortModelName(orch.Model())
	sessionName := sessions.Current()
	if sessionName == "" {
		sessionName = "default"
	}

	return fmt.Sprintf("serena:%s (%s) [%s] > ", project, sessionName, model)
}

func shortModelName(model string) string {
	trimmed := strings.TrimSpace(model)
	if trimmed == "" {
		return "model"
	}
	if idx := strings.LastIndex(trimmed, "/"); idx >= 0 && idx < len(trimmed)-1 {
		return trimmed[idx+1:]
	}
	return trimmed
}

func handleToolModeCommand(args []string, orch *orchestrator.Orchestrator) error {
	if len(args) == 0 {
		fmt.Printf("Tool mode: %s\n", orch.ToolMode())
		fmt.Println("Available: auto, guard, heuristic")
		return nil
	}

	mode := strings.ToLower(strings.TrimSpace(args[0]))
	if mode != "auto" && mode != "guard" && mode != "heuristic" {
		return fmt.Errorf("invalid tool mode: %s (use auto, guard, or heuristic)", mode)
	}
	orch.SetToolMode(mode)
	fmt.Printf("Tool mode set to %s\n", mode)
	return nil
}

func formatBanner(label string, value string, note string) string {
	if strings.TrimSpace(value) == "" {
		value = "-"
	}
	if note != "" {
		note = " " + note
	}
	if useColor() {
		return fmt.Sprintf(
			"%s%s%s %s%s%s%s%s%s\n",
			colorCyan,
			label+":",
			colorReset,
			colorGreen,
			value,
			colorReset,
			colorGray,
			note,
			colorReset,
		)
	}
	return fmt.Sprintf("%s: %s%s\n", label, value, note)
}

func handleContextImport(line string, orch *orchestrator.Orchestrator, sessions *SessionState) error {
	fields := strings.Fields(line)
	if len(fields) < 2 {
		return fmt.Errorf("usage: @context <path>")
	}
	path := strings.Join(fields[1:], " ")
	path = expandHome(path)

	data, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	if len(data) > maxContextFileSize {
		return fmt.Errorf("context file too large (%d bytes); limit is %d bytes", len(data), maxContextFileSize)
	}

	orch.AddContext(path, string(data))
	return sessions.SaveFromOrch(orch)
}

func expandHome(path string) string {
	if strings.HasPrefix(path, "~") {
		home, err := os.UserHomeDir()
		if err == nil {
			return filepath.Join(home, strings.TrimPrefix(path, "~"))
		}
	}
	return path
}
