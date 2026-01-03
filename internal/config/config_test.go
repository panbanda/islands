// Copyright 2025 Jon
// SPDX-License-Identifier: Apache-2.0

package config

import (
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestDefault(t *testing.T) {
	t.Parallel()

	cfg := Default()

	assert.NotEmpty(t, cfg.DataDir)
	assert.Equal(t, "info", cfg.LogLevel)

	assert.NotEmpty(t, cfg.Index.Exclude)
	assert.NotEmpty(t, cfg.Index.Include)
	assert.Equal(t, int64(1024*1024), cfg.Index.MaxFileSize)
	assert.Equal(t, 512, cfg.Index.ChunkSize)
	assert.Equal(t, 50, cfg.Index.ChunkOverlap)
	assert.Equal(t, 100, cfg.Index.BatchSize)
	assert.Equal(t, 4, cfg.Index.ConcurrentFiles)
	assert.Equal(t, 30*time.Minute, cfg.Index.Timeout)

	assert.Equal(t, "stdio", cfg.MCP.Transport)
	assert.Equal(t, "127.0.0.1", cfg.MCP.Host)
	assert.Equal(t, 8080, cfg.MCP.Port)
	assert.Equal(t, 50, cfg.MCP.MaxResults)
	assert.Equal(t, 5*time.Minute, cfg.MCP.RequestTimeout)
	assert.True(t, cfg.MCP.EnableResources)

	assert.NotEmpty(t, cfg.Git.CacheDir)
	assert.Equal(t, 10*time.Minute, cfg.Git.Timeout)
	assert.Len(t, cfg.Git.Providers, 3)

	assert.Equal(t, "python3", cfg.LEANN.PythonPath)
	assert.Equal(t, "all-MiniLM-L6-v2", cfg.LEANN.ModelName)
	assert.NotEmpty(t, cfg.LEANN.CacheDir)
	assert.Equal(t, "cpu", cfg.LEANN.DeviceType)
	assert.Equal(t, 32, cfg.LEANN.BatchSize)
	assert.Equal(t, 512, cfg.LEANN.MaxTokens)
	assert.True(t, cfg.LEANN.Compression)
	assert.Equal(t, 0.3, cfg.LEANN.PruningFactor)
}

func TestDefaultExcludePatterns(t *testing.T) {
	t.Parallel()

	cfg := Default()
	excludePatterns := cfg.Index.Exclude

	expectedPatterns := []string{
		"**/node_modules/**",
		"**/.git/**",
		"**/vendor/**",
	}

	for _, expected := range expectedPatterns {
		assert.Contains(t, excludePatterns, expected)
	}
}

func TestDefaultIncludePatterns(t *testing.T) {
	t.Parallel()

	cfg := Default()
	includePatterns := cfg.Index.Include

	expectedPatterns := []string{
		"**/*.go",
		"**/*.py",
		"**/*.js",
		"**/*.ts",
	}

	for _, expected := range expectedPatterns {
		assert.Contains(t, includePatterns, expected)
	}
}

func TestDefaultGitProviders(t *testing.T) {
	t.Parallel()

	cfg := Default()

	require.Len(t, cfg.Git.Providers, 3)

	github := cfg.Git.Providers[0]
	assert.Equal(t, "github", github.Name)
	assert.Equal(t, "github", github.Type)
	assert.Equal(t, "https://github.com", github.BaseURL)

	gitlab := cfg.Git.Providers[1]
	assert.Equal(t, "gitlab", gitlab.Name)
	assert.Equal(t, "gitlab", gitlab.Type)
	assert.Equal(t, "https://gitlab.com", gitlab.BaseURL)

	bitbucket := cfg.Git.Providers[2]
	assert.Equal(t, "bitbucket", bitbucket.Name)
	assert.Equal(t, "bitbucket", bitbucket.Type)
	assert.Equal(t, "https://bitbucket.org", bitbucket.BaseURL)
}

func TestConfigValidate(t *testing.T) {
	t.Parallel()

	cfg := Default()
	err := cfg.Validate()
	assert.NoError(t, err)
}

func TestLoadWithEnvironmentTokens(t *testing.T) {
	tempDir := t.TempDir()
	os.Setenv("PYTHIA_TEST_TOKEN", "secret-token")
	defer os.Unsetenv("PYTHIA_TEST_TOKEN")

	cfg := &Config{
		DataDir: tempDir,
		Git: GitConfig{
			Providers: []GitProvider{
				{
					Name:     "test",
					Type:     "github",
					TokenEnv: "PYTHIA_TEST_TOKEN",
				},
			},
		},
	}

	for i, p := range cfg.Git.Providers {
		if p.Token == "" && p.TokenEnv != "" {
			cfg.Git.Providers[i].Token = os.Getenv(p.TokenEnv)
		}
	}

	assert.Equal(t, "secret-token", cfg.Git.Providers[0].Token)
}

func TestDefaultDataDir(t *testing.T) {
	t.Parallel()

	cfg := Default()

	home, err := os.UserHomeDir()
	require.NoError(t, err)

	expected := filepath.Join(home, ".pythia")
	assert.Equal(t, expected, cfg.DataDir)
}

func TestIndexConfigDefaults(t *testing.T) {
	t.Parallel()

	cfg := Default()

	assert.Greater(t, len(cfg.Index.Exclude), 5)
	assert.Greater(t, len(cfg.Index.Include), 10)
}

func TestMCPConfigDefaults(t *testing.T) {
	t.Parallel()

	cfg := Default()

	assert.Equal(t, "stdio", cfg.MCP.Transport)
	assert.NotEqual(t, 0, cfg.MCP.Port)
	assert.NotEmpty(t, cfg.MCP.Host)
}

func TestLEANNConfigDefaults(t *testing.T) {
	t.Parallel()

	cfg := Default()

	assert.NotEmpty(t, cfg.LEANN.PythonPath)
	assert.NotEmpty(t, cfg.LEANN.ModelName)
	assert.NotEmpty(t, cfg.LEANN.DeviceType)
	assert.Greater(t, cfg.LEANN.BatchSize, 0)
	assert.Greater(t, cfg.LEANN.MaxTokens, 0)
	assert.Greater(t, cfg.LEANN.PruningFactor, 0.0)
	assert.Less(t, cfg.LEANN.PruningFactor, 1.0)
}
