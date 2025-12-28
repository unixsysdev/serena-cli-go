package main

import (
	"bufio"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"strconv"
	"strings"

	"github.com/unixsysdev/serena-cli-go/internal/config"
	"github.com/unixsysdev/serena-cli-go/internal/orchestrator"
)

var version = "dev"

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

	ctx := context.Background()

	if flag.NArg() > 0 {
		prompt := strings.Join(flag.Args(), " ")
		resp, err := orch.Chat(ctx, prompt)
		if err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		fmt.Println(resp)
		return
	}

	if err := runREPL(ctx, orch, cfg); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func runREPL(ctx context.Context, orch *orchestrator.Orchestrator, cfg *config.Config) error {
	scanner := bufio.NewScanner(os.Stdin)
	scanner.Buffer(make([]byte, 0, 64*1024), 1024*1024)

	fmt.Print("> ")
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			fmt.Print("> ")
			continue
		}
		if strings.HasPrefix(line, "/") {
			exit, err := handleCommand(line, orch, cfg)
			if err != nil {
				fmt.Println(err)
			}
			if exit {
				return nil
			}
			fmt.Print("> ")
			continue
		}
		if line == "exit" || line == "quit" {
			return nil
		}

		resp, err := orch.Chat(ctx, line)
		if err != nil {
			return err
		}

		fmt.Println(resp)
		fmt.Print("> ")
	}

	if err := scanner.Err(); err != nil {
		return err
	}

	return nil
}

func handleCommand(line string, orch *orchestrator.Orchestrator, cfg *config.Config) (bool, error) {
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
		return false, handleModelCommand(cmd, args, orch)
	case "config":
		return false, printConfig(cfg)
	case "reset":
		orch.Reset()
		fmt.Println("Conversation reset.")
		return false, nil
	default:
		return false, fmt.Errorf("unknown command: %s (try /help)", fields[0])
	}
}

func handleModelCommand(cmd string, args []string, orch *orchestrator.Orchestrator) error {
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
		fmt.Printf("Model set to %s\n", model)
		return nil
	}

	for _, model := range availableModels {
		if strings.EqualFold(model, arg) {
			orch.SetModel(model)
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
	fmt.Println("  /config         Show resolved config (API key masked)")
	fmt.Println("  /reset          Clear the conversation context")
	fmt.Println("  /exit, /quit    Exit the CLI")
}

func printConfig(cfg *config.Config) error {
	display := map[string]interface{}{
		"glm": map[string]string{
			"api_key":  maskKey(cfg.GLM.APIKey),
			"base_url": cfg.GLM.BaseURL,
			"model":    cfg.GLM.Model,
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
