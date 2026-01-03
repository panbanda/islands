// Copyright 2025 Jon
// SPDX-License-Identifier: Apache-2.0

package types

import (
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
)

func TestCodeChunk(t *testing.T) {
	t.Parallel()

	chunk := CodeChunk{
		ID:         "chunk-1",
		Content:    "func main() {}",
		File:       "/path/to/main.go",
		StartLine:  10,
		EndLine:    20,
		Language:   "go",
		Repository: "my-repo",
		Metadata: map[string]string{
			"package": "main",
		},
	}

	assert.Equal(t, "chunk-1", chunk.ID)
	assert.Equal(t, "func main() {}", chunk.Content)
	assert.Equal(t, "/path/to/main.go", chunk.File)
	assert.Equal(t, 10, chunk.StartLine)
	assert.Equal(t, 20, chunk.EndLine)
	assert.Equal(t, "go", chunk.Language)
	assert.Equal(t, "my-repo", chunk.Repository)
	assert.Equal(t, "main", chunk.Metadata["package"])
}

func TestFileInfo(t *testing.T) {
	t.Parallel()

	now := time.Now()
	info := FileInfo{
		Path:         "/abs/path/to/file.go",
		RelativePath: "file.go",
		Size:         1024,
		Language:     "go",
		LineCount:    50,
		ModifiedAt:   now,
		Checksum:     "abc123",
	}

	assert.Equal(t, "/abs/path/to/file.go", info.Path)
	assert.Equal(t, "file.go", info.RelativePath)
	assert.Equal(t, int64(1024), info.Size)
	assert.Equal(t, "go", info.Language)
	assert.Equal(t, 50, info.LineCount)
	assert.Equal(t, now, info.ModifiedAt)
	assert.Equal(t, "abc123", info.Checksum)
}

func TestQueryResult(t *testing.T) {
	t.Parallel()

	result := QueryResult{
		Query: "test query",
		Results: []SearchMatch{
			{File: "file1.go", Score: 0.9},
			{File: "file2.go", Score: 0.8},
		},
		TotalCount: 100,
		Duration:   500 * time.Millisecond,
	}

	assert.Equal(t, "test query", result.Query)
	assert.Len(t, result.Results, 2)
	assert.Equal(t, 100, result.TotalCount)
	assert.Equal(t, 500*time.Millisecond, result.Duration)
}

func TestSearchMatch(t *testing.T) {
	t.Parallel()

	match := SearchMatch{
		File:       "/path/to/file.go",
		Line:       42,
		Column:     10,
		Score:      0.95,
		Preview:    "func TestExample()",
		Context:    "surrounding code",
		Repository: "my-repo",
		Metadata: map[string]string{
			"function": "TestExample",
		},
	}

	assert.Equal(t, "/path/to/file.go", match.File)
	assert.Equal(t, 42, match.Line)
	assert.Equal(t, 10, match.Column)
	assert.Equal(t, 0.95, match.Score)
	assert.Equal(t, "func TestExample()", match.Preview)
	assert.Equal(t, "surrounding code", match.Context)
	assert.Equal(t, "my-repo", match.Repository)
	assert.Equal(t, "TestExample", match.Metadata["function"])
}

func TestLanguageStats(t *testing.T) {
	t.Parallel()

	stats := LanguageStats{
		Language:   "Go",
		FileCount:  100,
		LineCount:  5000,
		ByteCount:  150000,
		Percentage: 45.5,
	}

	assert.Equal(t, "Go", stats.Language)
	assert.Equal(t, 100, stats.FileCount)
	assert.Equal(t, 5000, stats.LineCount)
	assert.Equal(t, int64(150000), stats.ByteCount)
	assert.Equal(t, 45.5, stats.Percentage)
}

func TestRepositoryStats(t *testing.T) {
	t.Parallel()

	now := time.Now()
	stats := RepositoryStats{
		Name:       "my-repo",
		Path:       "/path/to/repo",
		FileCount:  250,
		LineCount:  15000,
		ByteCount:  500000,
		ChunkCount: 300,
		Languages: []LanguageStats{
			{Language: "Go", FileCount: 100, Percentage: 40.0},
			{Language: "Python", FileCount: 150, Percentage: 60.0},
		},
		IndexedAt:     now,
		IndexDuration: 5 * time.Second,
	}

	assert.Equal(t, "my-repo", stats.Name)
	assert.Equal(t, "/path/to/repo", stats.Path)
	assert.Equal(t, 250, stats.FileCount)
	assert.Equal(t, 15000, stats.LineCount)
	assert.Equal(t, int64(500000), stats.ByteCount)
	assert.Equal(t, 300, stats.ChunkCount)
	assert.Len(t, stats.Languages, 2)
	assert.Equal(t, now, stats.IndexedAt)
	assert.Equal(t, 5*time.Second, stats.IndexDuration)
}

func TestCodeChunkWithoutMetadata(t *testing.T) {
	t.Parallel()

	chunk := CodeChunk{
		ID:      "minimal",
		Content: "content",
	}

	assert.Equal(t, "minimal", chunk.ID)
	assert.Nil(t, chunk.Metadata)
}

func TestSearchMatchWithoutMetadata(t *testing.T) {
	t.Parallel()

	match := SearchMatch{
		File:  "file.go",
		Score: 0.5,
	}

	assert.Equal(t, "file.go", match.File)
	assert.Nil(t, match.Metadata)
}

func TestEmptyQueryResult(t *testing.T) {
	t.Parallel()

	result := QueryResult{
		Query:      "no results",
		Results:    []SearchMatch{},
		TotalCount: 0,
	}

	assert.Empty(t, result.Results)
	assert.Equal(t, 0, result.TotalCount)
}

func TestEmptyRepositoryStats(t *testing.T) {
	t.Parallel()

	stats := RepositoryStats{
		Name: "empty-repo",
	}

	assert.Equal(t, "empty-repo", stats.Name)
	assert.Empty(t, stats.Languages)
	assert.Zero(t, stats.FileCount)
}
