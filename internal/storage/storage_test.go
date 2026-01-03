// Copyright 2025 Jon
// SPDX-License-Identifier: Apache-2.0

package storage

import (
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestOpen(t *testing.T) {
	t.Parallel()

	tempDir := t.TempDir()

	store, err := Open(tempDir)
	require.NoError(t, err)
	require.NotNil(t, store)

	defer store.Close()

	indexDir := filepath.Join(tempDir, "indexes")
	info, err := os.Stat(indexDir)
	require.NoError(t, err)
	assert.True(t, info.IsDir())
}

func TestOpenCreatesDirectories(t *testing.T) {
	t.Parallel()

	tempDir := t.TempDir()
	dataDir := filepath.Join(tempDir, "nested", "path", "data")

	store, err := Open(dataDir)
	require.NoError(t, err)
	defer store.Close()

	info, err := os.Stat(dataDir)
	require.NoError(t, err)
	assert.True(t, info.IsDir())
}

func TestCreateAndGetIndex(t *testing.T) {
	t.Parallel()

	store := createTestStore(t)
	defer store.Close()

	info := IndexInfo{
		Name:      "test-index",
		Path:      "/path/to/code",
		FileCount: 100,
		Size:      1024000,
		Checksum:  "abc123",
	}

	err := store.CreateIndex(info)
	require.NoError(t, err)

	retrieved, err := store.GetIndex("test-index")
	require.NoError(t, err)
	require.NotNil(t, retrieved)

	assert.Equal(t, info.Name, retrieved.Name)
	assert.Equal(t, info.Path, retrieved.Path)
	assert.Equal(t, info.FileCount, retrieved.FileCount)
	assert.Equal(t, info.Size, retrieved.Size)
	assert.Equal(t, info.Checksum, retrieved.Checksum)
	assert.False(t, retrieved.CreatedAt.IsZero())
	assert.False(t, retrieved.UpdatedAt.IsZero())
}

func TestCreateIndexAlreadyExists(t *testing.T) {
	t.Parallel()

	store := createTestStore(t)
	defer store.Close()

	info := IndexInfo{
		Name:      "duplicate",
		Path:      "/path/to/code",
		FileCount: 50,
	}

	err := store.CreateIndex(info)
	require.NoError(t, err)

	err = store.CreateIndex(info)
	assert.ErrorIs(t, err, ErrAlreadyExists)
}

func TestGetIndexNotFound(t *testing.T) {
	t.Parallel()

	store := createTestStore(t)
	defer store.Close()

	_, err := store.GetIndex("nonexistent")
	assert.ErrorIs(t, err, ErrNotFound)
}

func TestUpdateIndex(t *testing.T) {
	t.Parallel()

	store := createTestStore(t)
	defer store.Close()

	original := IndexInfo{
		Name:      "update-test",
		Path:      "/original/path",
		FileCount: 100,
		Size:      1000,
	}

	err := store.CreateIndex(original)
	require.NoError(t, err)

	time.Sleep(10 * time.Millisecond)

	updated := IndexInfo{
		Name:      "update-test",
		Path:      "/original/path",
		FileCount: 200,
		Size:      2000,
	}

	err = store.UpdateIndex(updated)
	require.NoError(t, err)

	retrieved, err := store.GetIndex("update-test")
	require.NoError(t, err)

	assert.Equal(t, 200, retrieved.FileCount)
	assert.Equal(t, int64(2000), retrieved.Size)
	assert.True(t, retrieved.UpdatedAt.After(retrieved.CreatedAt))
}

func TestUpdateIndexNotFound(t *testing.T) {
	t.Parallel()

	store := createTestStore(t)
	defer store.Close()

	info := IndexInfo{
		Name: "nonexistent",
	}

	err := store.UpdateIndex(info)
	assert.ErrorIs(t, err, ErrNotFound)
}

func TestDeleteIndex(t *testing.T) {
	t.Parallel()

	store := createTestStore(t)
	defer store.Close()

	info := IndexInfo{
		Name:      "delete-test",
		Path:      "/path/to/delete",
		FileCount: 50,
	}

	err := store.CreateIndex(info)
	require.NoError(t, err)

	err = store.DeleteIndex("delete-test")
	require.NoError(t, err)

	_, err = store.GetIndex("delete-test")
	assert.ErrorIs(t, err, ErrNotFound)
}

func TestDeleteIndexNotFound(t *testing.T) {
	t.Parallel()

	store := createTestStore(t)
	defer store.Close()

	err := store.DeleteIndex("nonexistent")
	assert.ErrorIs(t, err, ErrNotFound)
}

func TestListIndexes(t *testing.T) {
	t.Parallel()

	store := createTestStore(t)
	defer store.Close()

	indexes := []IndexInfo{
		{Name: "index-1", Path: "/path/1", FileCount: 10},
		{Name: "index-2", Path: "/path/2", FileCount: 20},
		{Name: "index-3", Path: "/path/3", FileCount: 30},
	}

	for _, idx := range indexes {
		err := store.CreateIndex(idx)
		require.NoError(t, err)
	}

	list, err := store.ListIndexes()
	require.NoError(t, err)
	assert.Len(t, list, 3)

	names := make(map[string]bool)
	for _, idx := range list {
		names[idx.Name] = true
	}

	assert.True(t, names["index-1"])
	assert.True(t, names["index-2"])
	assert.True(t, names["index-3"])
}

func TestListIndexesEmpty(t *testing.T) {
	t.Parallel()

	store := createTestStore(t)
	defer store.Close()

	list, err := store.ListIndexes()
	require.NoError(t, err)
	assert.Empty(t, list)
}

func TestIndexPath(t *testing.T) {
	t.Parallel()

	tempDir := t.TempDir()
	store, err := Open(tempDir)
	require.NoError(t, err)
	defer store.Close()

	expected := filepath.Join(tempDir, "indexes", "my-index")
	assert.Equal(t, expected, store.IndexPath("my-index"))
}

func TestDataDir(t *testing.T) {
	t.Parallel()

	tempDir := t.TempDir()
	store, err := Open(tempDir)
	require.NoError(t, err)
	defer store.Close()

	assert.Equal(t, tempDir, store.DataDir())
}

func TestClose(t *testing.T) {
	t.Parallel()

	store := createTestStore(t)

	err := store.Close()
	assert.NoError(t, err)
}

func TestConcurrentAccess(t *testing.T) {
	t.Parallel()

	store := createTestStore(t)
	defer store.Close()

	done := make(chan bool)

	for i := 0; i < 10; i++ {
		go func(n int) {
			info := IndexInfo{
				Name:      "concurrent-" + string(rune('0'+n)),
				Path:      "/path",
				FileCount: n,
			}
			_ = store.CreateIndex(info)
			done <- true
		}(i)
	}

	for i := 0; i < 10; i++ {
		<-done
	}

	list, err := store.ListIndexes()
	require.NoError(t, err)
	assert.LessOrEqual(t, len(list), 10)
}

func createTestStore(t *testing.T) *Store {
	t.Helper()

	tempDir := t.TempDir()
	store, err := Open(tempDir)
	require.NoError(t, err)

	return store
}
