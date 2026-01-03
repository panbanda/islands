// Copyright 2025 Jon
// SPDX-License-Identifier: Apache-2.0

// Package types provides common types used across Pythia.
package types

import "time"

// CodeChunk represents a chunk of code for indexing.
type CodeChunk struct {
	ID         string            `json:"id"`
	Content    string            `json:"content"`
	File       string            `json:"file"`
	StartLine  int               `json:"startLine"`
	EndLine    int               `json:"endLine"`
	Language   string            `json:"language"`
	Repository string            `json:"repository"`
	Metadata   map[string]string `json:"metadata,omitempty"`
}

// FileInfo contains metadata about a source file.
type FileInfo struct {
	Path         string    `json:"path"`
	RelativePath string    `json:"relativePath"`
	Size         int64     `json:"size"`
	Language     string    `json:"language"`
	LineCount    int       `json:"lineCount"`
	ModifiedAt   time.Time `json:"modifiedAt"`
	Checksum     string    `json:"checksum"`
}

// QueryResult contains results from a query operation.
type QueryResult struct {
	Query      string         `json:"query"`
	Results    []SearchMatch  `json:"results"`
	TotalCount int            `json:"totalCount"`
	Duration   time.Duration  `json:"duration"`
}

// SearchMatch represents a single search match.
type SearchMatch struct {
	File       string            `json:"file"`
	Line       int               `json:"line"`
	Column     int               `json:"column"`
	Score      float64           `json:"score"`
	Preview    string            `json:"preview"`
	Context    string            `json:"context"`
	Repository string            `json:"repository"`
	Metadata   map[string]string `json:"metadata,omitempty"`
}

// LanguageStats contains statistics about a programming language.
type LanguageStats struct {
	Language   string `json:"language"`
	FileCount  int    `json:"fileCount"`
	LineCount  int    `json:"lineCount"`
	ByteCount  int64  `json:"byteCount"`
	Percentage float64 `json:"percentage"`
}

// RepositoryStats contains statistics about an indexed repository.
type RepositoryStats struct {
	Name         string          `json:"name"`
	Path         string          `json:"path"`
	FileCount    int             `json:"fileCount"`
	LineCount    int             `json:"lineCount"`
	ByteCount    int64           `json:"byteCount"`
	ChunkCount   int             `json:"chunkCount"`
	Languages    []LanguageStats `json:"languages"`
	IndexedAt    time.Time       `json:"indexedAt"`
	IndexDuration time.Duration  `json:"indexDuration"`
}
