// Copyright 2025 Jon
// SPDX-License-Identifier: Apache-2.0

// Package analyzer provides semantic search and analysis functionality.
package analyzer

import (
	"context"
	"fmt"
	"path/filepath"
	"strings"

	"github.com/jon/pythia/internal/storage"
	"github.com/jon/pythia/pkg/leann"
)

// Options configures the analyzer.
type Options struct {
	DataDir  string
	LogLevel string
}

// SearchOptions configures a search operation.
type SearchOptions struct {
	Query     string
	Index     string
	Limit     int
	Threshold float64
}

// SearchResult represents a single search result.
type SearchResult struct {
	File    string  `json:"file"`
	Line    int     `json:"line"`
	Score   float64 `json:"score"`
	Preview string  `json:"preview"`
	Index   string  `json:"index"`
}

// Analyzer provides semantic search over indexed codebases.
type Analyzer struct {
	opts  Options
	store *storage.Store
	leann *leann.Client
}

// New creates a new Analyzer.
func New(opts Options) (*Analyzer, error) {
	store, err := storage.Open(opts.DataDir)
	if err != nil {
		return nil, fmt.Errorf("failed to open storage: %w", err)
	}

	leannClient, err := leann.NewClient(leann.ClientOptions{
		CacheDir: filepath.Join(opts.DataDir, "leann-cache"),
	})
	if err != nil {
		store.Close()
		return nil, fmt.Errorf("failed to create LEANN client: %w", err)
	}

	return &Analyzer{
		opts:  opts,
		store: store,
		leann: leannClient,
	}, nil
}

// Close closes the analyzer.
func (a *Analyzer) Close() error {
	if err := a.leann.Close(); err != nil {
		return err
	}
	return a.store.Close()
}

// Search performs semantic search across indexed codebases.
func (a *Analyzer) Search(ctx context.Context, opts SearchOptions) ([]SearchResult, error) {
	if opts.Limit <= 0 {
		opts.Limit = 10
	}
	if opts.Threshold <= 0 {
		opts.Threshold = 0.5
	}

	var results []leann.SearchResult
	var err error

	if opts.Index != "" {
		results, err = a.leann.SearchIndex(ctx, opts.Index, leann.SearchOptions{
			Query:     opts.Query,
			Limit:     opts.Limit,
			Threshold: opts.Threshold,
		})
	} else {
		results, err = a.leann.Search(ctx, leann.SearchOptions{
			Query:     opts.Query,
			Limit:     opts.Limit,
			Threshold: opts.Threshold,
		})
	}
	if err != nil {
		return nil, fmt.Errorf("search failed: %w", err)
	}

	var searchResults []SearchResult
	for _, r := range results {
		searchResults = append(searchResults, SearchResult{
			File:    r.Metadata["file"],
			Line:    1,
			Score:   r.Score,
			Preview: truncatePreview(r.Content, 200),
			Index:   r.Metadata["index"],
		})
	}

	return searchResults, nil
}

// Ask performs a question-answering query over indexed codebases.
func (a *Analyzer) Ask(ctx context.Context, question string, indexName string) (string, error) {
	if indexName != "" {
		result, err := a.leann.Ask(ctx, indexName, question, 5)
		if err == nil && result != nil {
			var response strings.Builder
			response.WriteString("Relevant code snippets:\n\n")
			response.WriteString(result.Context)
			response.WriteString("\n\nSources:\n")
			for _, src := range result.Sources {
				response.WriteString(fmt.Sprintf("- %s (score: %.2f)\n", src.File, src.Score))
			}
			return response.String(), nil
		}
	}

	searchResults, err := a.Search(ctx, SearchOptions{
		Query:     question,
		Index:     indexName,
		Limit:     5,
		Threshold: 0.3,
	})
	if err != nil {
		return "", err
	}

	if len(searchResults) == 0 {
		return "No relevant code found for your question.", nil
	}

	var contextBuilder strings.Builder
	contextBuilder.WriteString("Relevant code snippets:\n\n")
	for i, r := range searchResults {
		contextBuilder.WriteString(fmt.Sprintf("--- %s (%.2f) ---\n%s\n\n", r.File, r.Score, r.Preview))
		if i >= 4 {
			break
		}
	}

	return contextBuilder.String(), nil
}

// GetIndexes returns all available indexes.
func (a *Analyzer) GetIndexes() ([]storage.IndexInfo, error) {
	return a.store.ListIndexes()
}

// GetIndex returns information about a specific index.
func (a *Analyzer) GetIndex(name string) (*storage.IndexInfo, error) {
	return a.store.GetIndex(name)
}

func truncatePreview(s string, maxLen int) string {
	s = strings.TrimSpace(s)
	lines := strings.Split(s, "\n")

	var result strings.Builder
	for _, line := range lines {
		if result.Len() > 0 {
			result.WriteString(" ")
		}
		result.WriteString(strings.TrimSpace(line))
		if result.Len() >= maxLen {
			break
		}
	}

	str := result.String()
	if len(str) > maxLen {
		return str[:maxLen-3] + "..."
	}
	return str
}
