// Copyright 2025 Jon
// SPDX-License-Identifier: Apache-2.0

// Package storage provides persistent storage for Pythia indexes.
package storage

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"time"
)

var (
	// ErrNotFound is returned when an index is not found.
	ErrNotFound = errors.New("index not found")
	// ErrAlreadyExists is returned when an index already exists.
	ErrAlreadyExists = errors.New("index already exists")
)

// IndexInfo holds metadata about an indexed codebase.
type IndexInfo struct {
	Name      string    `json:"name"`
	Path      string    `json:"path"`
	FileCount int       `json:"fileCount"`
	Size      int64     `json:"size"`
	CreatedAt time.Time `json:"createdAt"`
	UpdatedAt time.Time `json:"updatedAt"`
	Checksum  string    `json:"checksum"`
}

// Store provides persistent storage for indexes.
type Store struct {
	dataDir string
	mu      sync.RWMutex
}

// Open opens or creates a storage at the given directory.
func Open(dataDir string) (*Store, error) {
	if err := os.MkdirAll(dataDir, 0755); err != nil {
		return nil, fmt.Errorf("failed to create data directory: %w", err)
	}

	indexDir := filepath.Join(dataDir, "indexes")
	if err := os.MkdirAll(indexDir, 0755); err != nil {
		return nil, fmt.Errorf("failed to create index directory: %w", err)
	}

	return &Store{dataDir: dataDir}, nil
}

// Close closes the storage.
func (s *Store) Close() error {
	return nil
}

// ListIndexes returns all indexed codebases.
func (s *Store) ListIndexes() ([]IndexInfo, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	indexDir := filepath.Join(s.dataDir, "indexes")
	entries, err := os.ReadDir(indexDir)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, fmt.Errorf("failed to read index directory: %w", err)
	}

	var indexes []IndexInfo
	for _, entry := range entries {
		if !entry.IsDir() {
			continue
		}

		info, err := s.GetIndex(entry.Name())
		if err != nil {
			continue
		}
		indexes = append(indexes, *info)
	}

	return indexes, nil
}

// GetIndex returns metadata for a specific index.
func (s *Store) GetIndex(name string) (*IndexInfo, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	metaPath := filepath.Join(s.dataDir, "indexes", name, "meta.json")
	data, err := os.ReadFile(metaPath)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, ErrNotFound
		}
		return nil, fmt.Errorf("failed to read index metadata: %w", err)
	}

	var info IndexInfo
	if err := json.Unmarshal(data, &info); err != nil {
		return nil, fmt.Errorf("failed to parse index metadata: %w", err)
	}

	return &info, nil
}

// CreateIndex creates a new index entry.
func (s *Store) CreateIndex(info IndexInfo) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	indexPath := filepath.Join(s.dataDir, "indexes", info.Name)

	if _, err := os.Stat(indexPath); err == nil {
		return ErrAlreadyExists
	}

	if err := os.MkdirAll(indexPath, 0755); err != nil {
		return fmt.Errorf("failed to create index directory: %w", err)
	}

	info.CreatedAt = time.Now()
	info.UpdatedAt = info.CreatedAt

	return s.writeMetadata(info)
}

// UpdateIndex updates an existing index entry.
func (s *Store) UpdateIndex(info IndexInfo) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	indexPath := filepath.Join(s.dataDir, "indexes", info.Name)
	if _, err := os.Stat(indexPath); os.IsNotExist(err) {
		return ErrNotFound
	}

	info.UpdatedAt = time.Now()
	return s.writeMetadata(info)
}

// DeleteIndex removes an index.
func (s *Store) DeleteIndex(name string) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	indexPath := filepath.Join(s.dataDir, "indexes", name)
	if _, err := os.Stat(indexPath); os.IsNotExist(err) {
		return ErrNotFound
	}

	return os.RemoveAll(indexPath)
}

// IndexPath returns the filesystem path for an index.
func (s *Store) IndexPath(name string) string {
	return filepath.Join(s.dataDir, "indexes", name)
}

// DataDir returns the storage data directory.
func (s *Store) DataDir() string {
	return s.dataDir
}

func (s *Store) writeMetadata(info IndexInfo) error {
	metaPath := filepath.Join(s.dataDir, "indexes", info.Name, "meta.json")
	data, err := json.MarshalIndent(info, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal index metadata: %w", err)
	}

	if err := os.WriteFile(metaPath, data, 0644); err != nil {
		return fmt.Errorf("failed to write index metadata: %w", err)
	}

	return nil
}
