// Copyright 2025 Jon
// SPDX-License-Identifier: Apache-2.0

package leann

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestNewClient(t *testing.T) {
	t.Parallel()

	tempDir := t.TempDir()

	client, err := NewClient(ClientOptions{
		CacheDir: tempDir,
	})
	require.NoError(t, err)
	require.NotNil(t, client)

	defer client.Close()
}

func TestNewClientDefaults(t *testing.T) {
	t.Parallel()

	client, err := NewClient(ClientOptions{})
	require.NoError(t, err)
	require.NotNil(t, client)

	defer client.Close()

	assert.Equal(t, "all-MiniLM-L6-v2", client.opts.ModelName)
	assert.Equal(t, 32, client.opts.BatchSize)
}

func TestAddDocument(t *testing.T) {
	t.Parallel()

	client := createTestClient(t)
	defer client.Close()

	ctx := context.Background()
	err := client.AddDocument(ctx, Document{
		ID:      "doc1",
		Content: "This is test content",
		Metadata: map[string]string{
			"file": "test.go",
		},
	})

	require.NoError(t, err)
	assert.Equal(t, 1, client.Count())
}

func TestAddDocuments(t *testing.T) {
	t.Parallel()

	client := createTestClient(t)
	defer client.Close()

	ctx := context.Background()
	docs := []Document{
		{ID: "doc1", Content: "Content 1"},
		{ID: "doc2", Content: "Content 2"},
		{ID: "doc3", Content: "Content 3"},
	}

	err := client.AddDocuments(ctx, docs)
	require.NoError(t, err)
	assert.Equal(t, 3, client.Count())
}

func TestSearch(t *testing.T) {
	t.Parallel()

	client := createTestClient(t)
	defer client.Close()

	ctx := context.Background()

	docs := []Document{
		{ID: "doc1", Content: "Authentication middleware for HTTP"},
		{ID: "doc2", Content: "Database connection pooling"},
		{ID: "doc3", Content: "User login and session management"},
	}

	err := client.AddDocuments(ctx, docs)
	require.NoError(t, err)

	results, err := client.Search(ctx, SearchOptions{
		Query:     "authentication",
		Limit:     10,
		Threshold: 0.5,
	})

	require.NoError(t, err)
	assert.NotNil(t, results)
}

func TestSearchWithLimit(t *testing.T) {
	t.Parallel()

	client := createTestClient(t)
	defer client.Close()

	ctx := context.Background()

	for i := 0; i < 20; i++ {
		err := client.AddDocument(ctx, Document{
			ID:      string(rune('a' + i)),
			Content: "Test content",
		})
		require.NoError(t, err)
	}

	results, err := client.Search(ctx, SearchOptions{
		Query:     "test",
		Limit:     5,
		Threshold: 0.5,
	})

	require.NoError(t, err)
	assert.LessOrEqual(t, len(results), 5)
}

func TestSearchThreshold(t *testing.T) {
	t.Parallel()

	client := createTestClient(t)
	defer client.Close()

	ctx := context.Background()

	err := client.AddDocument(ctx, Document{
		ID:      "doc1",
		Content: "Very specific content about golang",
	})
	require.NoError(t, err)

	results, err := client.Search(ctx, SearchOptions{
		Query:     "python",
		Limit:     10,
		Threshold: 0.99,
	})

	require.NoError(t, err)
	assert.Empty(t, results)
}

func TestDeleteDocument(t *testing.T) {
	t.Parallel()

	client := createTestClient(t)
	defer client.Close()

	ctx := context.Background()

	err := client.AddDocument(ctx, Document{
		ID:      "to-delete",
		Content: "This will be deleted",
	})
	require.NoError(t, err)
	assert.Equal(t, 1, client.Count())

	err = client.DeleteDocument(ctx, "to-delete")
	require.NoError(t, err)
	assert.Equal(t, 0, client.Count())
}

func TestClear(t *testing.T) {
	t.Parallel()

	client := createTestClient(t)
	defer client.Close()

	ctx := context.Background()

	for i := 0; i < 5; i++ {
		err := client.AddDocument(ctx, Document{
			ID:      string(rune('a' + i)),
			Content: "Content",
		})
		require.NoError(t, err)
	}

	assert.Equal(t, 5, client.Count())

	err := client.Clear(ctx)
	require.NoError(t, err)
	assert.Equal(t, 0, client.Count())
}

func TestCount(t *testing.T) {
	t.Parallel()

	client := createTestClient(t)
	defer client.Close()

	ctx := context.Background()

	assert.Equal(t, 0, client.Count())

	for i := 0; i < 10; i++ {
		err := client.AddDocument(ctx, Document{
			ID:      string(rune('a' + i)),
			Content: "Content",
		})
		require.NoError(t, err)
		assert.Equal(t, i+1, client.Count())
	}
}

func TestClose(t *testing.T) {
	t.Parallel()

	client := createTestClient(t)

	err := client.Close()
	assert.NoError(t, err)
}

func TestDocument(t *testing.T) {
	t.Parallel()

	doc := Document{
		ID:      "test-id",
		Content: "Test content",
		Metadata: map[string]string{
			"file":  "test.go",
			"index": "my-index",
		},
	}

	assert.Equal(t, "test-id", doc.ID)
	assert.Equal(t, "Test content", doc.Content)
	assert.Equal(t, "test.go", doc.Metadata["file"])
	assert.Equal(t, "my-index", doc.Metadata["index"])
}

func TestSearchOptions(t *testing.T) {
	t.Parallel()

	opts := SearchOptions{
		Query:     "test query",
		Limit:     20,
		Threshold: 0.8,
	}

	assert.Equal(t, "test query", opts.Query)
	assert.Equal(t, 20, opts.Limit)
	assert.Equal(t, 0.8, opts.Threshold)
}

func TestSearchResult(t *testing.T) {
	t.Parallel()

	result := SearchResult{
		ID:      "result-id",
		Content: "Result content",
		Score:   0.95,
		Metadata: map[string]string{
			"file": "result.go",
		},
	}

	assert.Equal(t, "result-id", result.ID)
	assert.Equal(t, "Result content", result.Content)
	assert.Equal(t, 0.95, result.Score)
	assert.Equal(t, "result.go", result.Metadata["file"])
}

func TestClientOptions(t *testing.T) {
	t.Parallel()

	opts := ClientOptions{
		ServiceURL:    "http://localhost:8081",
		CacheDir:      "/cache",
		ModelName:     "custom-model",
		DeviceType:    "cuda",
		BatchSize:     64,
		Compression:   true,
		PruningFactor: 0.5,
	}

	assert.Equal(t, "http://localhost:8081", opts.ServiceURL)
	assert.Equal(t, "/cache", opts.CacheDir)
	assert.Equal(t, "custom-model", opts.ModelName)
	assert.Equal(t, "cuda", opts.DeviceType)
	assert.Equal(t, 64, opts.BatchSize)
	assert.True(t, opts.Compression)
	assert.Equal(t, 0.5, opts.PruningFactor)
}

func TestConcurrentAccess(t *testing.T) {
	t.Parallel()

	client := createTestClient(t)
	defer client.Close()

	ctx := context.Background()
	done := make(chan bool)

	for i := 0; i < 10; i++ {
		go func(n int) {
			doc := Document{
				ID:      string(rune('a' + n)),
				Content: "Concurrent content",
			}
			_ = client.AddDocument(ctx, doc)
			done <- true
		}(i)
	}

	for i := 0; i < 10; i++ {
		<-done
	}

	assert.LessOrEqual(t, client.Count(), 10)
}

func createTestClient(t *testing.T) *Client {
	t.Helper()

	tempDir := t.TempDir()
	client, err := NewClient(ClientOptions{
		CacheDir: tempDir,
	})
	require.NoError(t, err)

	return client
}
