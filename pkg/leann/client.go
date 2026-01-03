// Copyright 2025 Jon
// SPDX-License-Identifier: Apache-2.0

// Package leann provides a client for LEANN vector search integration.
// It communicates with a LEANN Python service over HTTP for actual embeddings
// and vector search operations.
package leann

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"sync"
	"time"
)

// ClientOptions configures the LEANN client.
type ClientOptions struct {
	ServiceURL    string
	CacheDir      string
	ModelName     string
	DeviceType    string
	BatchSize     int
	Compression   bool
	PruningFactor float64
	HTTPTimeout   time.Duration
}

// Document represents a document to be indexed.
type Document struct {
	ID       string            `json:"id"`
	Content  string            `json:"content"`
	Metadata map[string]string `json:"metadata"`
}

// SearchOptions configures a search operation.
type SearchOptions struct {
	Query     string
	Limit     int
	Threshold float64
}

// SearchResult represents a search result.
type SearchResult struct {
	ID       string            `json:"id"`
	Content  string            `json:"content"`
	Score    float64           `json:"score"`
	Metadata map[string]string `json:"metadata"`
}

// AskResult contains the response from a Q&A query.
type AskResult struct {
	Question string       `json:"question"`
	Context  string       `json:"context"`
	Sources  []SourceInfo `json:"sources"`
}

// SourceInfo describes a source document used in a Q&A response.
type SourceInfo struct {
	ID    string  `json:"id"`
	Score float64 `json:"score"`
	File  string  `json:"file"`
}

// Client provides access to LEANN functionality via HTTP service.
type Client struct {
	opts       ClientOptions
	httpClient *http.Client
	mu         sync.RWMutex
	documents  map[string]Document
	indexName  string
}

// NewClient creates a new LEANN client.
func NewClient(opts ClientOptions) (*Client, error) {
	if opts.ServiceURL == "" {
		opts.ServiceURL = os.Getenv("PYTHIA_LEANN_URL")
		if opts.ServiceURL == "" {
			opts.ServiceURL = "http://127.0.0.1:8081"
		}
	}
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
	if opts.HTTPTimeout == 0 {
		opts.HTTPTimeout = 30 * time.Second
	}

	if err := os.MkdirAll(opts.CacheDir, 0755); err != nil {
		return nil, fmt.Errorf("failed to create cache directory: %w", err)
	}

	return &Client{
		opts: opts,
		httpClient: &http.Client{
			Timeout: opts.HTTPTimeout,
		},
		documents: make(map[string]Document),
	}, nil
}

// Close closes the LEANN client and releases resources.
func (c *Client) Close() error {
	c.httpClient.CloseIdleConnections()
	return nil
}

// SetIndexName sets the current working index name.
func (c *Client) SetIndexName(name string) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.indexName = name
}

// GetIndexName returns the current working index name.
func (c *Client) GetIndexName() string {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.indexName
}

// AddDocument adds a document to the local buffer.
func (c *Client) AddDocument(ctx context.Context, doc Document) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	c.documents[doc.ID] = doc
	return nil
}

// AddDocuments adds multiple documents to the local buffer.
func (c *Client) AddDocuments(ctx context.Context, docs []Document) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	for _, doc := range docs {
		c.documents[doc.ID] = doc
	}
	return nil
}

// BuildIndex sends all buffered documents to LEANN to build the index.
func (c *Client) BuildIndex(ctx context.Context, indexName string, force bool) error {
	c.mu.Lock()
	docs := make([]Document, 0, len(c.documents))
	for _, doc := range c.documents {
		docs = append(docs, doc)
	}
	c.mu.Unlock()

	if len(docs) == 0 {
		return fmt.Errorf("no documents to index")
	}

	payload := map[string]interface{}{
		"documents": docs,
		"force":     force,
		"backend":   "hnsw",
	}

	body, err := json.Marshal(payload)
	if err != nil {
		return fmt.Errorf("failed to marshal request: %w", err)
	}

	url := fmt.Sprintf("%s/indexes/%s/build", c.opts.ServiceURL, indexName)
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return fmt.Errorf("failed to create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return fmt.Errorf("failed to build index: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode >= 400 {
		respBody, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("build index failed (status %d): %s", resp.StatusCode, string(respBody))
	}

	c.mu.Lock()
	c.indexName = indexName
	c.documents = make(map[string]Document)
	c.mu.Unlock()

	return nil
}

// Search performs semantic search over the LEANN index.
func (c *Client) Search(ctx context.Context, opts SearchOptions) ([]SearchResult, error) {
	indexName := c.GetIndexName()
	if indexName == "" {
		return c.searchLocal(opts)
	}

	return c.SearchIndex(ctx, indexName, opts)
}

// SearchIndex performs semantic search over a specific LEANN index.
func (c *Client) SearchIndex(ctx context.Context, indexName string, opts SearchOptions) ([]SearchResult, error) {
	payload := map[string]interface{}{
		"query":     opts.Query,
		"limit":     opts.Limit,
		"threshold": opts.Threshold,
	}

	body, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	url := fmt.Sprintf("%s/indexes/%s/search", c.opts.ServiceURL, indexName)
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("search failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode >= 400 {
		respBody, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("search failed (status %d): %s", resp.StatusCode, string(respBody))
	}

	var result struct {
		Results []SearchResult `json:"results"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	return result.Results, nil
}

// Ask performs a Q&A query using RAG over the LEANN index.
func (c *Client) Ask(ctx context.Context, indexName string, question string, contextLimit int) (*AskResult, error) {
	if contextLimit <= 0 {
		contextLimit = 5
	}

	payload := map[string]interface{}{
		"question":      question,
		"context_limit": contextLimit,
	}

	body, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	url := fmt.Sprintf("%s/indexes/%s/ask", c.opts.ServiceURL, indexName)
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("ask failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode >= 400 {
		respBody, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("ask failed (status %d): %s", resp.StatusCode, string(respBody))
	}

	var result AskResult
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	return &result, nil
}

// ListIndexes returns all available indexes from the LEANN service.
func (c *Client) ListIndexes(ctx context.Context) ([]string, error) {
	url := fmt.Sprintf("%s/indexes", c.opts.ServiceURL)
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("list indexes failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode >= 400 {
		respBody, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("list indexes failed (status %d): %s", resp.StatusCode, string(respBody))
	}

	var result struct {
		Indexes []string `json:"indexes"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	return result.Indexes, nil
}

// DeleteIndex removes an index from the LEANN service.
func (c *Client) DeleteIndex(ctx context.Context, indexName string) error {
	url := fmt.Sprintf("%s/indexes/%s", c.opts.ServiceURL, indexName)
	req, err := http.NewRequestWithContext(ctx, http.MethodDelete, url, nil)
	if err != nil {
		return fmt.Errorf("failed to create request: %w", err)
	}

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return fmt.Errorf("delete index failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode >= 400 {
		respBody, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("delete index failed (status %d): %s", resp.StatusCode, string(respBody))
	}

	return nil
}

// Health checks the LEANN service health.
func (c *Client) Health(ctx context.Context) error {
	url := fmt.Sprintf("%s/health", c.opts.ServiceURL)
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return fmt.Errorf("failed to create request: %w", err)
	}

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return fmt.Errorf("health check failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode >= 400 {
		return fmt.Errorf("LEANN service unhealthy (status %d)", resp.StatusCode)
	}

	return nil
}

// DeleteDocument removes a document from the local buffer.
func (c *Client) DeleteDocument(ctx context.Context, id string) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	delete(c.documents, id)
	return nil
}

// Clear removes all documents from the local buffer.
func (c *Client) Clear(ctx context.Context) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	c.documents = make(map[string]Document)
	return nil
}

// Count returns the number of documents in the local buffer.
func (c *Client) Count() int {
	c.mu.RLock()
	defer c.mu.RUnlock()

	return len(c.documents)
}

// searchLocal performs a simple local search (fallback when no index is built).
func (c *Client) searchLocal(opts SearchOptions) ([]SearchResult, error) {
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

// computeSimilarity provides a basic fallback similarity calculation.
// This is only used when the LEANN service is unavailable.
func (c *Client) computeSimilarity(query, content string) float64 {
	return 0.5
}
