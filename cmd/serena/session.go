package main

import (
	"bufio"
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/sashabaranov/go-openai"
	"github.com/unixsysdev/serena-cli-go/internal/config"
	"github.com/unixsysdev/serena-cli-go/internal/orchestrator"
	"github.com/unixsysdev/serena-cli-go/internal/session"
)

const (
	defaultSessionName = "default"
)

type SessionState struct {
	store   *session.Store
	data    *session.SessionData
	name    string
	baseDir string
}

func initSessionState(cfg *config.Config, orch *orchestrator.Orchestrator) (*SessionState, error) {
	baseDir, err := sessionBaseDir(cfg)
	if err != nil {
		return nil, err
	}

	store, err := session.NewStore(baseDir)
	if err != nil {
		return nil, err
	}

	state := &SessionState{
		store:   store,
		baseDir: baseDir,
	}

	if err := state.loadOrCreate(defaultSessionName, orch); err != nil {
		return nil, err
	}

	registerSessionTools(orch, state)
	return state, nil
}

func (s *SessionState) Current() string {
	return s.name
}

func (s *SessionState) SaveFromOrch(orch *orchestrator.Orchestrator) error {
	if s.data == nil {
		return nil
	}
	s.data.Model = orch.Model()
	s.data.SystemPrompt = orch.SystemPrompt()
	s.data.Messages = session.FromOpenAIMessages(orch.Messages())
	return s.store.Save(s.data)
}

func (s *SessionState) loadOrCreate(name string, orch *orchestrator.Orchestrator) error {
	sessionName := sanitizeSessionName(name)
	data, err := s.store.Load(sessionName)
	if err != nil {
		if !errors.Is(err, os.ErrNotExist) {
			return err
		}
		data = &session.SessionData{
			Name:         sessionName,
			Model:        orch.Model(),
			SystemPrompt: orch.SystemPrompt(),
			ArchiveFile:  sessionName + "_archive.txt",
			SummaryFile:  sessionName + "_summary.md",
		}
		if err := s.store.Save(data); err != nil {
			return err
		}
	}

	s.name = sessionName
	s.data = data

	if s.data.Model != "" && s.data.Model != orch.Model() {
		orch.SetModel(s.data.Model)
	}

	messages := session.ToOpenAIMessages(orch.SystemPrompt(), s.data.Messages)
	orch.ReplaceMessages(messages)
	return nil
}

func (s *SessionState) Switch(name string, orch *orchestrator.Orchestrator) error {
	sessionName := sanitizeSessionName(name)
	if sessionName == "" {
		return fmt.Errorf("session name required")
	}
	if sessionName == s.name {
		return nil
	}
	return s.loadOrCreate(sessionName, orch)
}

func (s *SessionState) Delete(name string) error {
	sessionName := sanitizeSessionName(name)
	if sessionName == "" {
		return fmt.Errorf("session name required")
	}
	if sessionName == s.name {
		return fmt.Errorf("cannot delete active session")
	}
	return s.store.Delete(sessionName)
}

func (s *SessionState) ArchivePath() string {
	if s.data == nil || s.data.ArchiveFile == "" {
		return ""
	}
	return filepath.Join(s.baseDir, s.data.ArchiveFile)
}

func (s *SessionState) SummaryPath() string {
	if s.data == nil || s.data.SummaryFile == "" {
		return ""
	}
	return filepath.Join(s.baseDir, s.data.SummaryFile)
}

func (s *SessionState) AppendArchive(content string) error {
	if content == "" {
		return nil
	}
	path := s.ArchivePath()
	if path == "" {
		return nil
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	f, err := os.OpenFile(path, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o600)
	if err != nil {
		return err
	}
	defer f.Close()

	header := fmt.Sprintf("\n---\nCompaction at %s\n---\n", time.Now().Format(time.RFC3339))
	if _, err := f.WriteString(header + content + "\n"); err != nil {
		return err
	}
	return nil
}

func (s *SessionState) WriteSummary(content string) error {
	path := s.SummaryPath()
	if path == "" {
		return nil
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	return os.WriteFile(path, []byte(content), 0o600)
}

func (s *SessionState) SearchArchive(query string, maxResults int) (string, error) {
	path := s.ArchivePath()
	if path == "" {
		return "No archive file for this session.", nil
	}
	file, err := os.Open(path)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return "No archive content yet.", nil
		}
		return "", err
	}
	defer file.Close()

	queryLower := strings.ToLower(query)
	if queryLower == "" {
		return "Query is empty.", nil
	}
	if maxResults <= 0 {
		maxResults = 10
	}

	var matches []string
	scanner := bufio.NewScanner(file)
	lineNo := 0
	for scanner.Scan() {
		lineNo++
		line := scanner.Text()
		if strings.Contains(strings.ToLower(line), queryLower) {
			matches = append(matches, fmt.Sprintf("%d: %s", lineNo, line))
			if len(matches) >= maxResults {
				break
			}
		}
	}
	if err := scanner.Err(); err != nil {
		return "", err
	}

	if len(matches) == 0 {
		return "No matches found in session archive.", nil
	}
	return strings.Join(matches, "\n"), nil
}

func handleSessionCommand(args []string, orch *orchestrator.Orchestrator, sessions *SessionState, ui *ConsoleUI) error {
	if len(args) == 0 || args[0] == "list" {
		return printSessions(sessions)
	}

	switch args[0] {
	case "new":
		if len(args) < 2 {
			return fmt.Errorf("usage: /session new <name>")
		}
		if err := sessions.Switch(args[1], orch); err != nil {
			return err
		}
		ui.StopSpinner()
		return sessions.SaveFromOrch(orch)
	case "switch":
		if len(args) < 2 {
			return fmt.Errorf("usage: /session switch <name>")
		}
		if err := sessions.Switch(args[1], orch); err != nil {
			return err
		}
		ui.StopSpinner()
		return sessions.SaveFromOrch(orch)
	case "delete":
		if len(args) < 2 {
			return fmt.Errorf("usage: /session delete <name>")
		}
		return sessions.Delete(args[1])
	case "info":
		return printSessionInfo(sessions)
	default:
		return fmt.Errorf("unknown session command (use list/new/switch/delete/info)")
	}
}

func printSessions(sessions *SessionState) error {
	all, err := sessions.store.List()
	if err != nil {
		return err
	}
	if len(all) == 0 {
		fmt.Println("No sessions found.")
		return nil
	}
	fmt.Println("Sessions:")
	for _, entry := range all {
		marker := " "
		if entry.Name == sessions.name {
			marker = "*"
		}
		fmt.Printf("%s %s (updated %s)\n", marker, entry.Name, entry.UpdatedAt.Format(time.RFC822))
	}
	return nil
}

func printSessionInfo(sessions *SessionState) error {
	if sessions.data == nil {
		fmt.Println("No active session.")
		return nil
	}
	fmt.Printf("Session: %s\n", sessions.data.Name)
	fmt.Printf("Model: %s\n", sessions.data.Model)
	fmt.Printf("Updated: %s\n", sessions.data.UpdatedAt.Format(time.RFC822))
	if sessions.data.ArchiveFile != "" {
		fmt.Printf("Archive: %s\n", sessions.ArchivePath())
	}
	if sessions.data.SummaryFile != "" {
		fmt.Printf("Summary: %s\n", sessions.SummaryPath())
	}
	return nil
}

func compactSession(ctx context.Context, orch *orchestrator.Orchestrator, sessions *SessionState) error {
	messages := orch.Messages()
	if len(messages) < 4 {
		return fmt.Errorf("not enough messages to compact yet")
	}

	keep := 6
	if len(messages)-1 <= keep {
		return fmt.Errorf("not enough history to compact (need more than %d messages)", keep)
	}

	older := messages[1 : len(messages)-keep]
	recent := messages[len(messages)-keep:]

	transcript := buildTranscript(older)
	if transcript == "" {
		return fmt.Errorf("nothing to compact")
	}

	summary, err := orch.Summarize(ctx, transcript)
	if err != nil {
		return err
	}

	if err := sessions.AppendArchive(transcript); err != nil {
		return err
	}
	if err := sessions.WriteSummary(summary); err != nil {
		return err
	}

	archiveHint := sessions.ArchivePath()
	summaryMsg := openai.ChatCompletionMessage{
		Role: openai.ChatMessageRoleAssistant,
		Content: fmt.Sprintf("<summary>\n%s\n</summary>\n<archive>\n%s\nUse session_search to look up details.\n</archive>",
			strings.TrimSpace(summary), archiveHint),
	}

	newMessages := []openai.ChatCompletionMessage{
		messages[0],
		summaryMsg,
	}
	newMessages = append(newMessages, recent...)
	orch.ReplaceMessages(newMessages)

	if err := sessions.SaveFromOrch(orch); err != nil {
		return err
	}

	fmt.Println("Context compacted.")
	return nil
}

func buildTranscript(messages []openai.ChatCompletionMessage) string {
	var b strings.Builder
	for _, msg := range messages {
		role := msg.Role
		if role == "" {
			role = "unknown"
		}
		b.WriteString("[" + role + "]\n")
		if msg.Content != "" {
			b.WriteString(msg.Content)
			b.WriteString("\n")
		}
		if len(msg.ToolCalls) > 0 {
			for _, call := range msg.ToolCalls {
				line := fmt.Sprintf("tool_call: %s %s\n", call.Function.Name, call.Function.Arguments)
				b.WriteString(line)
			}
		}
		b.WriteString("\n")
	}
	return strings.TrimSpace(b.String())
}

func registerSessionTools(orch *orchestrator.Orchestrator, sessions *SessionState) {
	searchTool := openai.Tool{
		Type: openai.ToolTypeFunction,
		Function: &openai.FunctionDefinition{
			Name:        "session_search",
			Description: "Searches the compacted session archive for a query and returns matching lines.",
			Parameters: map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"query": map[string]interface{}{
						"type":        "string",
						"description": "Substring to search for in the session archive.",
					},
					"max_results": map[string]interface{}{
						"type":        "integer",
						"description": "Maximum number of matching lines to return.",
					},
				},
				"required": []string{"query"},
			},
		},
	}

	orch.AddLocalTool(searchTool, func(ctx context.Context, args map[string]interface{}) (string, error) {
		query, _ := args["query"].(string)
		maxResults := 10
		if raw, ok := args["max_results"]; ok {
			switch v := raw.(type) {
			case float64:
				maxResults = int(v)
			case int:
				maxResults = v
			case string:
				if parsed, err := strconv.Atoi(v); err == nil {
					maxResults = parsed
				}
			}
		}
		return sessions.SearchArchive(query, maxResults)
	})
}

func sessionBaseDir(cfg *config.Config) (string, error) {
	projectPath := cfg.Serena.ProjectPath
	if projectPath == "" || projectPath == "." {
		cwd, err := os.Getwd()
		if err != nil {
			return "", err
		}
		projectPath = cwd
	}
	absPath, err := filepath.Abs(projectPath)
	if err != nil {
		return "", err
	}
	projectName := sanitizeSessionName(filepath.Base(absPath))
	home, err := os.UserHomeDir()
	if err != nil {
		return "", err
	}
	return filepath.Join(home, ".serena-cli", "sessions", projectName), nil
}

func sanitizeSessionName(name string) string {
	trimmed := strings.TrimSpace(name)
	if trimmed == "" {
		return defaultSessionName
	}
	trimmed = strings.ToLower(trimmed)
	trimmed = strings.ReplaceAll(trimmed, " ", "-")
	trimmed = strings.Map(func(r rune) rune {
		switch {
		case r >= 'a' && r <= 'z':
			return r
		case r >= '0' && r <= '9':
			return r
		case r == '-' || r == '_' || r == '.':
			return r
		default:
			return -1
		}
	}, trimmed)
	if trimmed == "" {
		return defaultSessionName
	}
	return trimmed
}
