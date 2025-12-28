package session

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/sashabaranov/go-openai"
)

// StoredToolCall captures the minimal data needed to rebuild a tool call.
type StoredToolCall struct {
	ID        string `json:"id"`
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

// StoredMessage is a serializable representation of a chat message.
type StoredMessage struct {
	Role       string           `json:"role"`
	Content    string           `json:"content"`
	ToolCalls  []StoredToolCall `json:"tool_calls,omitempty"`
	ToolCallID string           `json:"tool_call_id,omitempty"`
}

// SessionData persists a conversation session.
type SessionData struct {
	Name         string          `json:"name"`
	CreatedAt    time.Time       `json:"created_at"`
	UpdatedAt    time.Time       `json:"updated_at"`
	Model        string          `json:"model"`
	SystemPrompt string          `json:"system_prompt"`
	Messages     []StoredMessage `json:"messages"`
	ArchiveFile  string          `json:"archive_file,omitempty"`
	SummaryFile  string          `json:"summary_file,omitempty"`
}

// Store manages session persistence in a directory.
type Store struct {
	dir string
}

// NewStore creates a new session store rooted at dir.
func NewStore(dir string) (*Store, error) {
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return nil, fmt.Errorf("create session dir: %w", err)
	}
	return &Store{dir: dir}, nil
}

// Path returns the session file path for name.
func (s *Store) Path(name string) string {
	return filepath.Join(s.dir, sanitizeName(name)+".json")
}

// Load reads a session by name.
func (s *Store) Load(name string) (*SessionData, error) {
	data, err := os.ReadFile(s.Path(name))
	if err != nil {
		return nil, err
	}

	var session SessionData
	if err := json.Unmarshal(data, &session); err != nil {
		return nil, fmt.Errorf("parse session: %w", err)
	}
	return &session, nil
}

// Save writes a session to disk.
func (s *Store) Save(session *SessionData) error {
	if session.Name == "" {
		return fmt.Errorf("session name is required")
	}
	if session.CreatedAt.IsZero() {
		session.CreatedAt = time.Now()
	}
	session.UpdatedAt = time.Now()

	payload, err := json.MarshalIndent(session, "", "  ")
	if err != nil {
		return fmt.Errorf("encode session: %w", err)
	}

	return os.WriteFile(s.Path(session.Name), payload, 0o600)
}

// Delete removes a session file.
func (s *Store) Delete(name string) error {
	return os.Remove(s.Path(name))
}

// List returns all stored sessions.
func (s *Store) List() ([]SessionData, error) {
	entries, err := os.ReadDir(s.dir)
	if err != nil {
		return nil, err
	}

	var sessions []SessionData
	for _, entry := range entries {
		if entry.IsDir() || !strings.HasSuffix(entry.Name(), ".json") {
			continue
		}
		data, err := os.ReadFile(filepath.Join(s.dir, entry.Name()))
		if err != nil {
			continue
		}
		var session SessionData
		if err := json.Unmarshal(data, &session); err != nil {
			continue
		}
		sessions = append(sessions, session)
	}

	sort.Slice(sessions, func(i, j int) bool {
		return sessions[i].UpdatedAt.After(sessions[j].UpdatedAt)
	})

	return sessions, nil
}

// FromOpenAIMessages converts OpenAI messages into stored messages, skipping the system prompt.
func FromOpenAIMessages(messages []openai.ChatCompletionMessage) []StoredMessage {
	if len(messages) == 0 {
		return nil
	}

	stored := make([]StoredMessage, 0, len(messages))
	for idx, msg := range messages {
		if idx == 0 && msg.Role == openai.ChatMessageRoleSystem {
			continue
		}

		entry := StoredMessage{
			Role:       msg.Role,
			Content:    msg.Content,
			ToolCallID: msg.ToolCallID,
		}

		if len(msg.ToolCalls) > 0 {
			entry.ToolCalls = make([]StoredToolCall, 0, len(msg.ToolCalls))
			for _, call := range msg.ToolCalls {
				entry.ToolCalls = append(entry.ToolCalls, StoredToolCall{
					ID:        call.ID,
					Name:      call.Function.Name,
					Arguments: call.Function.Arguments,
				})
			}
		}

		stored = append(stored, entry)
	}

	return stored
}

// ToOpenAIMessages rebuilds OpenAI messages using the provided system prompt.
func ToOpenAIMessages(systemPrompt string, messages []StoredMessage) []openai.ChatCompletionMessage {
	result := []openai.ChatCompletionMessage{
		{
			Role:    openai.ChatMessageRoleSystem,
			Content: systemPrompt,
		},
	}

	for _, msg := range messages {
		entry := openai.ChatCompletionMessage{
			Role:       msg.Role,
			Content:    msg.Content,
			ToolCallID: msg.ToolCallID,
		}

		if len(msg.ToolCalls) > 0 {
			entry.ToolCalls = make([]openai.ToolCall, 0, len(msg.ToolCalls))
			for _, call := range msg.ToolCalls {
				entry.ToolCalls = append(entry.ToolCalls, openai.ToolCall{
					ID:   call.ID,
					Type: openai.ToolTypeFunction,
					Function: openai.FunctionCall{
						Name:      call.Name,
						Arguments: call.Arguments,
					},
				})
			}
		}

		result = append(result, entry)
	}

	return result
}

func sanitizeName(name string) string {
	trimmed := strings.TrimSpace(name)
	if trimmed == "" {
		return "default"
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
		return "default"
	}
	return trimmed
}
