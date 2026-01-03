// Copyright 2025 Jon
// SPDX-License-Identifier: Apache-2.0

package indexer

import (
	"context"
	"net/http"
	"os"
	"path/filepath"
	"testing"
	"time"

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

	idx, err := New(Options{
		DataDir: tempDir,
	})
	require.NoError(t, err)
	require.NotNil(t, idx)

	defer idx.Close()
}

func TestIndex(t *testing.T) {
	t.Parallel()
	skipIfNoLEANN(t)

	tempDir := t.TempDir()
	codeDir := filepath.Join(tempDir, "code")

	createTestFiles(t, codeDir)

	idx, err := New(Options{
		DataDir: tempDir,
		Include: []string{"**/*.go", "**/*.txt"},
	})
	require.NoError(t, err)
	defer idx.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	var indexedFiles []string
	stats, err := idx.Index(ctx, IndexOptions{
		Name: "test-project",
		Path: codeDir,
		OnFile: func(path string) {
			indexedFiles = append(indexedFiles, path)
		},
	})

	require.NoError(t, err)
	require.NotNil(t, stats)

	assert.Greater(t, stats.Files, 0)
	assert.Greater(t, stats.Bytes, int64(0))
	assert.Greater(t, stats.Duration, time.Duration(0))
}

func TestIndexForceReindex(t *testing.T) {
	t.Parallel()
	skipIfNoLEANN(t)

	tempDir := t.TempDir()
	codeDir := filepath.Join(tempDir, "code")

	createTestFiles(t, codeDir)

	idx, err := New(Options{
		DataDir: tempDir,
		Include: []string{"**/*.go"},
	})
	require.NoError(t, err)
	defer idx.Close()

	ctx := context.Background()

	_, err = idx.Index(ctx, IndexOptions{
		Name: "force-test",
		Path: codeDir,
	})
	require.NoError(t, err)

	_, err = idx.Index(ctx, IndexOptions{
		Name: "force-test",
		Path: codeDir,
	})
	assert.Error(t, err)

	_, err = idx.Index(ctx, IndexOptions{
		Name:  "force-test",
		Path:  codeDir,
		Force: true,
	})
	assert.NoError(t, err)
}

func TestIndexEmptyDirectory(t *testing.T) {
	t.Parallel()

	tempDir := t.TempDir()
	emptyDir := filepath.Join(tempDir, "empty")
	err := os.MkdirAll(emptyDir, 0755)
	require.NoError(t, err)

	idx, err := New(Options{
		DataDir: tempDir,
		Include: []string{"**/*.go"},
	})
	require.NoError(t, err)
	defer idx.Close()

	ctx := context.Background()
	_, err = idx.Index(ctx, IndexOptions{
		Name: "empty",
		Path: emptyDir,
	})

	assert.Error(t, err)
	assert.Contains(t, err.Error(), "no files found")
}

func TestIndexNonexistentPath(t *testing.T) {
	t.Parallel()

	tempDir := t.TempDir()

	idx, err := New(Options{
		DataDir: tempDir,
	})
	require.NoError(t, err)
	defer idx.Close()

	ctx := context.Background()
	_, err = idx.Index(ctx, IndexOptions{
		Name: "nonexistent",
		Path: "/nonexistent/path",
	})

	assert.Error(t, err)
}

func TestIndexContextCancellation(t *testing.T) {
	t.Parallel()

	tempDir := t.TempDir()
	codeDir := filepath.Join(tempDir, "code")

	for i := 0; i < 100; i++ {
		dir := filepath.Join(codeDir, "pkg"+string(rune('0'+i%10)))
		os.MkdirAll(dir, 0755)
		content := "package main\n\nfunc main() {}\n"
		os.WriteFile(filepath.Join(dir, "file.go"), []byte(content), 0644)
	}

	idx, err := New(Options{
		DataDir: tempDir,
		Include: []string{"**/*.go"},
	})
	require.NoError(t, err)
	defer idx.Close()

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	_, err = idx.Index(ctx, IndexOptions{
		Name: "cancelled",
		Path: codeDir,
	})

	assert.Error(t, err)
}

func TestShouldExclude(t *testing.T) {
	t.Parallel()

	idx := &Indexer{
		opts: Options{
			Exclude: []string{
				"node_modules",
				".git",
				"vendor",
			},
		},
	}

	tests := []struct {
		path    string
		isDir   bool
		exclude bool
	}{
		{"node_modules", true, true},
		{".git", true, true},
		{"vendor", true, true},
		{"main.go", false, false},
		{"utils.js", false, false},
	}

	for _, tt := range tests {
		t.Run(tt.path, func(t *testing.T) {
			result := idx.shouldExclude(tt.path, tt.isDir)
			assert.Equal(t, tt.exclude, result)
		})
	}
}

func TestShouldInclude(t *testing.T) {
	t.Parallel()

	idx := &Indexer{
		opts: Options{
			Include: []string{
				"**/*.go",
				"**/*.py",
				"**/*.js",
			},
		},
	}

	tests := []struct {
		path    string
		include bool
	}{
		{"main.go", true},
		{"script.py", true},
		{"app.js", true},
		{"image.png", false},
		{"data.csv", false},
	}

	for _, tt := range tests {
		t.Run(tt.path, func(t *testing.T) {
			result := idx.shouldInclude(tt.path)
			assert.Equal(t, tt.include, result)
		})
	}
}

func TestShouldIncludeEmptyPatterns(t *testing.T) {
	t.Parallel()

	idx := &Indexer{
		opts: Options{
			Include: []string{},
		},
	}

	assert.True(t, idx.shouldInclude("anything.xyz"))
}

func TestChunkContent(t *testing.T) {
	t.Parallel()

	idx := &Indexer{}

	content := "line 1\nline 2\nline 3\nline 4\nline 5\n"
	chunks := idx.chunkContent(content)

	assert.NotEmpty(t, chunks)

	combined := ""
	for _, chunk := range chunks {
		if len(combined) == 0 {
			combined = chunk
		}
	}
	assert.Contains(t, combined, "line 1")
}

func TestChunkContentLong(t *testing.T) {
	t.Parallel()

	idx := &Indexer{}

	var lines string
	for i := 0; i < 200; i++ {
		lines += "This is a line of code that takes up some space in the file.\n"
	}

	chunks := idx.chunkContent(lines)
	assert.Greater(t, len(chunks), 1)
}

func TestComputeChecksum(t *testing.T) {
	t.Parallel()

	tempDir := t.TempDir()

	file1 := filepath.Join(tempDir, "file1.txt")
	file2 := filepath.Join(tempDir, "file2.txt")

	os.WriteFile(file1, []byte("content 1"), 0644)
	os.WriteFile(file2, []byte("content 2"), 0644)

	idx := &Indexer{}

	checksum1 := idx.computeChecksum([]string{file1, file2})
	checksum2 := idx.computeChecksum([]string{file1, file2})

	assert.Equal(t, checksum1, checksum2)
	assert.Len(t, checksum1, 16)

	os.WriteFile(file1, []byte("modified content"), 0644)
	checksum3 := idx.computeChecksum([]string{file1, file2})

	assert.NotEqual(t, checksum1, checksum3)
}

func TestClose(t *testing.T) {
	t.Parallel()

	tempDir := t.TempDir()

	idx, err := New(Options{
		DataDir: tempDir,
	})
	require.NoError(t, err)

	err = idx.Close()
	assert.NoError(t, err)
}

func createTestFiles(t *testing.T, dir string) {
	t.Helper()

	files := map[string]string{
		"main.go": `package main

func main() {
	println("Hello, World!")
}
`,
		"lib/utils.go": `package lib

func Add(a, b int) int {
	return a + b
}
`,
		"docs/readme.txt": "This is a test project.",
	}

	for path, content := range files {
		fullPath := filepath.Join(dir, path)
		err := os.MkdirAll(filepath.Dir(fullPath), 0755)
		require.NoError(t, err)
		err = os.WriteFile(fullPath, []byte(content), 0644)
		require.NoError(t, err)
	}
}
