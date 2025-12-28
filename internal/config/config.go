package config

import (
	"fmt"
	"os"
	"path/filepath"

	"github.com/spf13/viper"
)

// Config holds all configuration for Serena CLI
type Config struct {
	LLM       LLMConfig    `mapstructure:"llm"`
	LegacyLLM LLMConfig    `mapstructure:"glm"`
	Serena    SerenaConfig `mapstructure:"serena"`
	Debug     bool         `mapstructure:"debug"`
}

// LLMConfig holds LLM API configuration.
type LLMConfig struct {
	APIKey          string `mapstructure:"api_key"`
	BaseURL         string `mapstructure:"base_url"`
	Model           string `mapstructure:"model"`
	CompactionModel string `mapstructure:"compaction_model"`
}

// SerenaConfig holds Serena MCP configuration
type SerenaConfig struct {
	ProjectPath string            `mapstructure:"project_path"`
	Context     string            `mapstructure:"context"`
	Command     string            `mapstructure:"command"`
	Args        []string          `mapstructure:"args"`
	Env         map[string]string `mapstructure:"env"`
}

// LoadOptions controls configuration loading behavior.
type LoadOptions struct {
	SkipValidation bool
}

// Load loads configuration from file, environment, and defaults.
func Load() (*Config, error) {
	return LoadWithOptions(LoadOptions{})
}

// LoadWithOptions loads configuration with custom options.
func LoadWithOptions(opts LoadOptions) (*Config, error) {
	v := viper.New()

	// Set defaults
	setDefaults(v)

	// Config file search paths
	v.SetConfigName("serena-cli")
	v.SetConfigType("yaml")

	v.AddConfigPath(".")
	home, _ := os.UserHomeDir()
	v.AddConfigPath(filepath.Join(home, ".serena-cli"))
	v.AddConfigPath(filepath.Join(home, ".config", "serena-cli"))

	// Read config file (optional)
	if err := v.ReadInConfig(); err != nil {
		if _, ok := err.(viper.ConfigFileNotFoundError); !ok {
			return nil, fmt.Errorf("failed to read config: %w", err)
		}
	}

	// Environment variables
	v.SetEnvPrefix("SERENA")
	v.AutomaticEnv()

	// Direct LLM env vars
	v.BindEnv("llm.api_key", "LLM_API_KEY", "CHUTES_API_KEY")
	v.BindEnv("llm.base_url", "LLM_BASE_URL", "CHUTES_BASE_URL")
	v.BindEnv("llm.model", "LLM_MODEL", "CHUTES_MODEL")
	v.BindEnv("llm.compaction_model", "LLM_COMPACTION_MODEL", "CHUTES_COMPACTION_MODEL")

	// Parse config.
	var cfg Config
	if err := v.Unmarshal(&cfg); err != nil {
		return nil, fmt.Errorf("failed to parse config: %w", err)
	}

	mergeLegacyConfig(&cfg)

	// Validate
	if !opts.SkipValidation {
		if err := Validate(&cfg); err != nil {
			return nil, err
		}
	}

	return &cfg, nil
}

// setDefaults sets default configuration values
func setDefaults(v *viper.Viper) {
	v.SetDefault("llm.base_url", "https://llm.chutes.ai/v1")
	v.SetDefault("llm.model", "zai-org/GLM-4.7-TEE")
	v.SetDefault("llm.compaction_model", "Qwen/Qwen3-VL-235B-A22B-Instruct")
	v.SetDefault("serena.context", "claude-desktop")
	v.SetDefault("serena.command", "uvx")
	v.SetDefault("serena.args", []string{
		"--from", "git+https://github.com/oraios/serena",
		"serena", "start-mcp-server",
	})
	v.SetDefault("debug", false)
}

// Validate validates the configuration
func Validate(cfg *Config) error {
	if cfg.LLM.APIKey == "" {
		return fmt.Errorf("LLM API key is required (set LLM_API_KEY or configure in serena-cli.yaml)")
	}
	return nil
}

func mergeLegacyConfig(cfg *Config) {
	if legacyEmpty(cfg.LegacyLLM) {
		return
	}

	if cfg.LLM.APIKey == "" {
		cfg.LLM.APIKey = cfg.LegacyLLM.APIKey
	}
	if cfg.LLM.BaseURL == "" {
		cfg.LLM.BaseURL = cfg.LegacyLLM.BaseURL
	}
	if cfg.LLM.Model == "" {
		cfg.LLM.Model = cfg.LegacyLLM.Model
	}
	if cfg.LLM.CompactionModel == "" {
		cfg.LLM.CompactionModel = cfg.LegacyLLM.CompactionModel
	}
}

func legacyEmpty(cfg LLMConfig) bool {
	return cfg.APIKey == "" && cfg.BaseURL == "" && cfg.Model == "" && cfg.CompactionModel == ""
}
