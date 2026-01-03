// Copyright 2025 Jon
// SPDX-License-Identifier: Apache-2.0

package analyzer

import (
	"context"
	"net/http"
	"os"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func skipIfNoLEANN(t *testing.T) {
	t.Helper()
	leannURL := os.Getenv("PYTHIA_LEANN_URL")
	if leannURL == "" {
		leannURL = "http://127.0.0.1:8081"
	}
	resp, err := http.Get(leannURL + "/health")
	if err != nil || resp.StatusCode != http.StatusOK {
		t.Skip("LEANN service not available, skipping integration test")
	}
	if resp != nil {
		resp.Body.Close()
	}
}

func TestNew(t *testing.T) {
	t.Parallel()

	tempDir := t.TempDir()

	a, err := New(Options{
		DataDir: tempDir,
	})
	require.NoError(t, err)
	require.NotNil(t, a)

	defer a.Close()
}

func TestSearch(t *testing.T) {
	t.Parallel()

	tempDir := t.TempDir()

	a, err := New(Options{
		DataDir: tempDir,
	})
	require.NoError(t, err)
	defer a.Close()

	ctx := context.Background()
	results, err := a.Search(ctx, SearchOptions{
		Query:     "authentication middleware",
		Limit:     10,
		Threshold: 0.5,
	})

	require.NoError(t, err)
	assert.Empty(t, results)
}

func TestSearchWithDefaults(t *testing.T) {
	t.Parallel()

	tempDir := t.TempDir()

	a, err := New(Options{
		DataDir: tempDir,
	})
	require.NoError(t, err)
	defer a.Close()

	ctx := context.Background()
	results, err := a.Search(ctx, SearchOptions{
		Query: "test query",
	})

	require.NoError(t, err)
	assert.Empty(t, results)
}

func TestSearchWithIndex(t *testing.T) {
	t.Parallel()
	skipIfNoLEANN(t)

	tempDir := t.TempDir()

	a, err := New(Options{
		DataDir: tempDir,
	})
	require.NoError(t, err)
	defer a.Close()

	ctx := context.Background()
	results, err := a.Search(ctx, SearchOptions{
		Query: "test query",
		Index: "specific-index",
		Limit: 5,
	})

	require.NoError(t, err)
	assert.Empty(t, results)
}

func TestAsk(t *testing.T) {
	t.Parallel()

	tempDir := t.TempDir()

	a, err := New(Options{
		DataDir: tempDir,
	})
	require.NoError(t, err)
	defer a.Close()

	ctx := context.Background()
	answer, err := a.Ask(ctx, "What does this function do?", "")

	require.NoError(t, err)
	assert.NotEmpty(t, answer)
}

func TestAskWithIndex(t *testing.T) {
	t.Parallel()
	skipIfNoLEANN(t)

	tempDir := t.TempDir()

	a, err := New(Options{
		DataDir: tempDir,
	})
	require.NoError(t, err)
	defer a.Close()

	ctx := context.Background()
	answer, err := a.Ask(ctx, "How is authentication handled?", "my-project")

	require.NoError(t, err)
	assert.NotEmpty(t, answer)
}

func TestGetIndexes(t *testing.T) {
	t.Parallel()

	tempDir := t.TempDir()

	a, err := New(Options{
		DataDir: tempDir,
	})
	require.NoError(t, err)
	defer a.Close()

	indexes, err := a.GetIndexes()
	require.NoError(t, err)
	assert.Empty(t, indexes)
}

func TestGetIndex(t *testing.T) {
	t.Parallel()

	tempDir := t.TempDir()

	a, err := New(Options{
		DataDir: tempDir,
	})
	require.NoError(t, err)
	defer a.Close()

	_, err = a.GetIndex("nonexistent")
	assert.Error(t, err)
}

func TestClose(t *testing.T) {
	t.Parallel()

	tempDir := t.TempDir()

	a, err := New(Options{
		DataDir: tempDir,
	})
	require.NoError(t, err)

	err = a.Close()
	assert.NoError(t, err)
}

func TestTruncatePreview(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name     string
		input    string
		maxLen   int
		expected string
	}{
		{
			name:     "short text",
			input:    "short",
			maxLen:   100,
			expected: "short",
		},
		{
			name:     "exact length",
			input:    "exact",
			maxLen:   5,
			expected: "exact",
		},
		{
			name:     "truncated",
			input:    "this is a long string that needs truncation",
			maxLen:   20,
			expected: "this is a long st...",
		},
		{
			name:     "multiline",
			input:    "line 1\nline 2\nline 3",
			maxLen:   100,
			expected: "line 1 line 2 line 3",
		},
		{
			name:     "with whitespace",
			input:    "  trimmed  \n  content  ",
			maxLen:   100,
			expected: "trimmed content",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := truncatePreview(tt.input, tt.maxLen)
			assert.Equal(t, tt.expected, result)
		})
	}
}

func TestSearchResult(t *testing.T) {
	t.Parallel()

	result := SearchResult{
		File:    "/path/to/file.go",
		Line:    42,
		Score:   0.95,
		Preview: "func main() {}",
		Index:   "my-project",
	}

	assert.Equal(t, "/path/to/file.go", result.File)
	assert.Equal(t, 42, result.Line)
	assert.Equal(t, 0.95, result.Score)
	assert.Equal(t, "func main() {}", result.Preview)
	assert.Equal(t, "my-project", result.Index)
}

func TestSearchOptions(t *testing.T) {
	t.Parallel()

	opts := SearchOptions{
		Query:     "test query",
		Index:     "my-index",
		Limit:     20,
		Threshold: 0.7,
	}

	assert.Equal(t, "test query", opts.Query)
	assert.Equal(t, "my-index", opts.Index)
	assert.Equal(t, 20, opts.Limit)
	assert.Equal(t, 0.7, opts.Threshold)
}

func TestOptions(t *testing.T) {
	t.Parallel()

	opts := Options{
		DataDir:  "/data",
		LogLevel: "debug",
	}

	assert.Equal(t, "/data", opts.DataDir)
	assert.Equal(t, "debug", opts.LogLevel)
}
