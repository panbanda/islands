// Copyright 2025 Jon
// SPDX-License-Identifier: Apache-2.0

// Package config provides configuration management for Pythia.
package config

import (
	"os"
	"path/filepath"
	"time"

	"github.com/spf13/viper"
)

// Config holds all configuration for Pythia.
type Config struct {
	DataDir  string      `mapstructure:"data_dir"`
	LogLevel string      `mapstructure:"log_level"`
	Index    IndexConfig `mapstructure:"index"`
	MCP      MCPConfig   `mapstructure:"mcp"`
	Git      GitConfig   `mapstructure:"git"`
	LEANN    LEANNConfig `mapstructure:"leann"`
}

// IndexConfig holds indexing configuration.
type IndexConfig struct {
	Exclude         []string      `mapstructure:"exclude"`
	Include         []string      `mapstructure:"include"`
	MaxFileSize     int64         `mapstructure:"max_file_size"`
	ChunkSize       int           `mapstructure:"chunk_size"`
	ChunkOverlap    int           `mapstructure:"chunk_overlap"`
	BatchSize       int           `mapstructure:"batch_size"`
	ConcurrentFiles int           `mapstructure:"concurrent_files"`
	Timeout         time.Duration `mapstructure:"timeout"`
}

// MCPConfig holds MCP server configuration.
type MCPConfig struct {
	Transport       string        `mapstructure:"transport"`
	Host            string        `mapstructure:"host"`
	Port            int           `mapstructure:"port"`
	MaxResults      int           `mapstructure:"max_results"`
	RequestTimeout  time.Duration `mapstructure:"request_timeout"`
	EnableResources bool          `mapstructure:"enable_resources"`
}

// GitConfig holds Git provider configuration.
type GitConfig struct {
	Providers []GitProvider `mapstructure:"providers"`
	CacheDir  string        `mapstructure:"cache_dir"`
	Timeout   time.Duration `mapstructure:"timeout"`
}

// GitProvider holds configuration for a Git provider.
type GitProvider struct {
	Name     string `mapstructure:"name"`
	Type     string `mapstructure:"type"`
	BaseURL  string `mapstructure:"base_url"`
	Token    string `mapstructure:"token"`
	TokenEnv string `mapstructure:"token_env"`
}

// LEANNConfig holds LEANN integration configuration.
type LEANNConfig struct {
	PythonPath    string `mapstructure:"python_path"`
	ModelName     string `mapstructure:"model_name"`
	CacheDir      string `mapstructure:"cache_dir"`
	DeviceType    string `mapstructure:"device_type"`
	BatchSize     int    `mapstructure:"batch_size"`
	MaxTokens     int    `mapstructure:"max_tokens"`
	Compression   bool   `mapstructure:"compression"`
	PruningFactor float64 `mapstructure:"pruning_factor"`
}

// Default returns a Config with default values.
func Default() *Config {
	home, _ := os.UserHomeDir()
	dataDir := filepath.Join(home, ".pythia")

	return &Config{
		DataDir:  dataDir,
		LogLevel: "info",
		Index: IndexConfig{
			Exclude: []string{
				"**/node_modules/**",
				"**/.git/**",
				"**/vendor/**",
				"**/__pycache__/**",
				"**/dist/**",
				"**/build/**",
				"**/*.min.js",
				"**/*.min.css",
				"**/package-lock.json",
				"**/yarn.lock",
				"**/go.sum",
			},
			Include: []string{
				"**/*.go",
				"**/*.py",
				"**/*.js",
				"**/*.ts",
				"**/*.jsx",
				"**/*.tsx",
				"**/*.java",
				"**/*.rs",
				"**/*.c",
				"**/*.cpp",
				"**/*.h",
				"**/*.hpp",
				"**/*.rb",
				"**/*.php",
				"**/*.cs",
				"**/*.md",
				"**/*.txt",
				"**/*.yaml",
				"**/*.yml",
				"**/*.json",
				"**/*.toml",
			},
			MaxFileSize:     1024 * 1024, // 1MB
			ChunkSize:       512,
			ChunkOverlap:    50,
			BatchSize:       100,
			ConcurrentFiles: 4,
			Timeout:         30 * time.Minute,
		},
		MCP: MCPConfig{
			Transport:       "stdio",
			Host:            "127.0.0.1",
			Port:            8080,
			MaxResults:      50,
			RequestTimeout:  5 * time.Minute,
			EnableResources: true,
		},
		Git: GitConfig{
			CacheDir: filepath.Join(dataDir, "git-cache"),
			Timeout:  10 * time.Minute,
			Providers: []GitProvider{
				{Name: "github", Type: "github", BaseURL: "https://github.com"},
				{Name: "gitlab", Type: "gitlab", BaseURL: "https://gitlab.com"},
				{Name: "bitbucket", Type: "bitbucket", BaseURL: "https://bitbucket.org"},
			},
		},
		LEANN: LEANNConfig{
			PythonPath:    "python3",
			ModelName:     "all-MiniLM-L6-v2",
			CacheDir:      filepath.Join(dataDir, "leann-cache"),
			DeviceType:    "cpu",
			BatchSize:     32,
			MaxTokens:     512,
			Compression:   true,
			PruningFactor: 0.3,
		},
	}
}

// Load loads configuration from Viper.
func Load() (*Config, error) {
	cfg := Default()

	if err := viper.Unmarshal(cfg); err != nil {
		return nil, err
	}

	if cfg.DataDir == "" {
		cfg.DataDir = Default().DataDir
	}

	for i, p := range cfg.Git.Providers {
		if p.Token == "" && p.TokenEnv != "" {
			cfg.Git.Providers[i].Token = os.Getenv(p.TokenEnv)
		}
	}

	return cfg, nil
}

// Validate validates the configuration.
func (c *Config) Validate() error {
	return nil
}
