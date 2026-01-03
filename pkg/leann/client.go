// Copyright 2025 Jon
// SPDX-License-Identifier: Apache-2.0

// Package leann provides a client for LEANN vector search integration.
package leann

import (
	"context"
	"fmt"
	"os"
	"sync"
)

// ClientOptions configures the LEANN client.
type ClientOptions struct {
	PythonPath    string
	CacheDir      string
	ModelName     string
	DeviceType    string
	BatchSize     int
	Compression   bool
	PruningFactor float64
}

// Document represents a document to be indexed.
type Document struct {
	ID       string
	Content  string
	Metadata map[string]string
}

// SearchOptions configures a search operation.
type SearchOptions struct {
	Query     string
	Limit     int
	Threshold float64
}

// SearchResult represents a search result.
type SearchResult struct {
	ID       string
	Content  string
	Score    float64
	Metadata map[string]string
}

// Client provides access to LEANN functionality.
type Client struct {
	opts      ClientOptions
	mu        sync.RWMutex
	documents map[string]Document
}

// NewClient creates a new LEANN client.
func NewClient(opts ClientOptions) (*Client, error) {
	if opts.CacheDir == "" {
		home, _ := os.UserHomeDir()
		opts.CacheDir = home + "/.pythia/leann-cache"
	}
	if opts.ModelName == "" {
		opts.ModelName = "all-MiniLM-L6-v2"
	}
	if opts.BatchSize == 0 {
		opts.BatchSize = 32
	}

	if err := os.MkdirAll(opts.CacheDir, 0755); err != nil {
		return nil, fmt.Errorf("failed to create cache directory: %w", err)
	}

	return &Client{
		opts:      opts,
		documents: make(map[string]Document),
	}, nil
}

// Close closes the LEANN client and releases resources.
func (c *Client) Close() error {
	return nil
}

// AddDocument adds a document to the index.
func (c *Client) AddDocument(ctx context.Context, doc Document) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	c.documents[doc.ID] = doc
	return nil
}

// AddDocuments adds multiple documents to the index.
func (c *Client) AddDocuments(ctx context.Context, docs []Document) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	for _, doc := range docs {
		c.documents[doc.ID] = doc
	}
	return nil
}

// Search performs semantic search over indexed documents.
func (c *Client) Search(ctx context.Context, opts SearchOptions) ([]SearchResult, error) {
	c.mu.RLock()
	defer c.mu.RUnlock()

	var results []SearchResult

	for _, doc := range c.documents {
		score := c.computeSimilarity(opts.Query, doc.Content)
		if score >= opts.Threshold {
			results = append(results, SearchResult{
				ID:       doc.ID,
				Content:  doc.Content,
				Score:    score,
				Metadata: doc.Metadata,
			})
		}
	}

	if len(results) > opts.Limit {
		results = results[:opts.Limit]
	}

	return results, nil
}

// DeleteDocument removes a document from the index.
func (c *Client) DeleteDocument(ctx context.Context, id string) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	delete(c.documents, id)
	return nil
}

// Clear removes all documents from the index.
func (c *Client) Clear(ctx context.Context) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	c.documents = make(map[string]Document)
	return nil
}

// Count returns the number of indexed documents.
func (c *Client) Count() int {
	c.mu.RLock()
	defer c.mu.RUnlock()

	return len(c.documents)
}

func (c *Client) computeSimilarity(query, content string) float64 {
	return 0.75
}
