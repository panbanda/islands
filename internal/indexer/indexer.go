// Copyright 2025 Jon
// SPDX-License-Identifier: Apache-2.0

// Package indexer provides codebase indexing functionality.
package indexer

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io"
	"io/fs"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/jon/pythia/internal/storage"
	"github.com/jon/pythia/pkg/leann"
)

// Options configures the indexer.
type Options struct {
	DataDir  string
	Exclude  []string
	Include  []string
	LogLevel string
}

// IndexOptions configures a specific indexing operation.
type IndexOptions struct {
	Name   string
	Path   string
	IsGit  bool
	Force  bool
	OnFile func(path string)
}

// IndexStats contains statistics from an indexing operation.
type IndexStats struct {
	Files    int
	Bytes    int64
	Duration time.Duration
	Chunks   int
}

// Indexer indexes codebases for semantic search.
type Indexer struct {
	opts    Options
	store   *storage.Store
	leann   *leann.Client
	mu      sync.Mutex
}

// New creates a new Indexer.
func New(opts Options) (*Indexer, error) {
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

	return &Indexer{
		opts:  opts,
		store: store,
		leann: leannClient,
	}, nil
}

// Close closes the indexer and releases resources.
func (idx *Indexer) Close() error {
	if err := idx.leann.Close(); err != nil {
		return err
	}
	return idx.store.Close()
}

// Index indexes a codebase at the given path.
func (idx *Indexer) Index(ctx context.Context, opts IndexOptions) (*IndexStats, error) {
	idx.mu.Lock()
	defer idx.mu.Unlock()

	start := time.Now()

	path := opts.Path
	if opts.IsGit {
		var err error
		path, err = idx.cloneRepository(ctx, opts.Path)
		if err != nil {
			return nil, fmt.Errorf("failed to clone repository: %w", err)
		}
	}

	absPath, err := filepath.Abs(path)
	if err != nil {
		return nil, fmt.Errorf("failed to resolve path: %w", err)
	}

	info, err := os.Stat(absPath)
	if err != nil {
		return nil, fmt.Errorf("path does not exist: %w", err)
	}
	if !info.IsDir() {
		return nil, fmt.Errorf("path is not a directory: %s", absPath)
	}

	existing, err := idx.store.GetIndex(opts.Name)
	if err == nil && !opts.Force {
		return nil, fmt.Errorf("index %q already exists (use --force to overwrite)", opts.Name)
	}

	var files []string
	var totalBytes int64

	err = filepath.WalkDir(absPath, func(p string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}

		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}

		relPath, _ := filepath.Rel(absPath, p)

		if d.IsDir() {
			if idx.shouldExclude(relPath, true) {
				return filepath.SkipDir
			}
			return nil
		}

		if idx.shouldExclude(relPath, false) {
			return nil
		}

		if !idx.shouldInclude(relPath) {
			return nil
		}

		info, err := d.Info()
		if err != nil {
			return nil
		}

		files = append(files, p)
		totalBytes += info.Size()

		if opts.OnFile != nil {
			opts.OnFile(relPath)
		}

		return nil
	})
	if err != nil {
		return nil, fmt.Errorf("failed to walk directory: %w", err)
	}

	if len(files) == 0 {
		return nil, fmt.Errorf("no files found to index")
	}

	chunks, err := idx.processFiles(ctx, opts.Name, files)
	if err != nil {
		return nil, fmt.Errorf("failed to process files: %w", err)
	}

	checksum := idx.computeChecksum(files)

	indexInfo := storage.IndexInfo{
		Name:      opts.Name,
		Path:      absPath,
		FileCount: len(files),
		Size:      totalBytes,
		Checksum:  checksum,
	}

	if existing != nil {
		indexInfo.CreatedAt = existing.CreatedAt
		err = idx.store.UpdateIndex(indexInfo)
	} else {
		err = idx.store.CreateIndex(indexInfo)
	}
	if err != nil {
		return nil, fmt.Errorf("failed to save index metadata: %w", err)
	}

	return &IndexStats{
		Files:    len(files),
		Bytes:    totalBytes,
		Duration: time.Since(start),
		Chunks:   chunks,
	}, nil
}

func (idx *Indexer) shouldExclude(path string, isDir bool) bool {
	for _, pattern := range idx.opts.Exclude {
		matched, _ := filepath.Match(filepath.Base(pattern), filepath.Base(path))
		if matched {
			return true
		}

		if strings.Contains(pattern, "**") {
			simplePattern := strings.ReplaceAll(pattern, "**"+string(filepath.Separator), "")
			simplePattern = strings.ReplaceAll(simplePattern, string(filepath.Separator)+"**", "")
			if matched, _ := filepath.Match(simplePattern, filepath.Base(path)); matched {
				return true
			}
		}
	}
	return false
}

func (idx *Indexer) shouldInclude(path string) bool {
	if len(idx.opts.Include) == 0 {
		return true
	}

	for _, pattern := range idx.opts.Include {
		simplePattern := strings.TrimPrefix(pattern, "**/")
		if matched, _ := filepath.Match(simplePattern, filepath.Base(path)); matched {
			return true
		}
	}
	return false
}

func (idx *Indexer) processFiles(ctx context.Context, indexName string, files []string) (int, error) {
	var totalChunks atomic.Int32
	var firstErr error
	var errOnce sync.Once

	sem := make(chan struct{}, 4)
	var wg sync.WaitGroup

	for _, file := range files {
		select {
		case <-ctx.Done():
			return int(totalChunks.Load()), ctx.Err()
		case sem <- struct{}{}:
		}

		wg.Add(1)
		go func(f string) {
			defer wg.Done()
			defer func() { <-sem }()

			chunks, err := idx.processFile(ctx, indexName, f)
			if err != nil {
				errOnce.Do(func() { firstErr = err })
				return
			}
			totalChunks.Add(int32(chunks))
		}(file)
	}

	wg.Wait()

	if firstErr != nil {
		return int(totalChunks.Load()), firstErr
	}

	if err := idx.leann.BuildIndex(ctx, indexName, true); err != nil {
		return int(totalChunks.Load()), fmt.Errorf("failed to build LEANN index: %w", err)
	}

	return int(totalChunks.Load()), nil
}

func (idx *Indexer) processFile(ctx context.Context, indexName, filePath string) (int, error) {
	content, err := os.ReadFile(filePath)
	if err != nil {
		return 0, fmt.Errorf("failed to read file: %w", err)
	}

	chunks := idx.chunkContent(string(content))

	for i, chunk := range chunks {
		err := idx.leann.AddDocument(ctx, leann.Document{
			ID:       fmt.Sprintf("%s:%s:%d", indexName, filePath, i),
			Content:  chunk,
			Metadata: map[string]string{"file": filePath, "index": indexName},
		})
		if err != nil {
			return i, fmt.Errorf("failed to add document: %w", err)
		}
	}

	return len(chunks), nil
}

func (idx *Indexer) chunkContent(content string) []string {
	const chunkSize = 512
	const overlap = 50

	lines := strings.Split(content, "\n")
	var chunks []string
	var current strings.Builder
	lineCount := 0

	for _, line := range lines {
		current.WriteString(line)
		current.WriteString("\n")
		lineCount++

		if current.Len() >= chunkSize {
			chunks = append(chunks, current.String())
			current.Reset()

			overlapStart := lineCount - 3
			if overlapStart < 0 {
				overlapStart = 0
			}
			for i := overlapStart; i < lineCount && i < len(lines); i++ {
				current.WriteString(lines[i])
				current.WriteString("\n")
			}
		}
	}

	if current.Len() > 0 {
		chunks = append(chunks, current.String())
	}

	return chunks
}

func (idx *Indexer) cloneRepository(ctx context.Context, url string) (string, error) {
	return "", fmt.Errorf("git clone not yet implemented")
}

func (idx *Indexer) computeChecksum(files []string) string {
	h := sha256.New()
	for _, f := range files {
		file, err := os.Open(f)
		if err != nil {
			continue
		}
		io.Copy(h, file)
		file.Close()
	}
	return hex.EncodeToString(h.Sum(nil))[:16]
}
