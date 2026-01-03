// Copyright 2025 Jon
// SPDX-License-Identifier: Apache-2.0

package mcp

import (
	"context"
	"testing"

	"github.com/mark3labs/mcp-go/mcp"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestNewServer(t *testing.T) {
	t.Parallel()

	tempDir := t.TempDir()

	server, err := NewServer(ServerOptions{
		DataDir:   tempDir,
		Transport: "stdio",
		LogLevel:  "info",
	})
	require.NoError(t, err)
	require.NotNil(t, server)
}

func TestNewServerHTTP(t *testing.T) {
	t.Parallel()

	tempDir := t.TempDir()

	server, err := NewServer(ServerOptions{
		DataDir:   tempDir,
		Transport: "http",
		Host:      "127.0.0.1",
		Port:      8080,
		LogLevel:  "info",
	})
	require.NoError(t, err)
	require.NotNil(t, server)
}

func TestServerOptions(t *testing.T) {
	t.Parallel()

	opts := ServerOptions{
		DataDir:   "/data",
		Transport: "http",
		Host:      "0.0.0.0",
		Port:      9090,
		LogLevel:  "debug",
	}

	assert.Equal(t, "/data", opts.DataDir)
	assert.Equal(t, "http", opts.Transport)
	assert.Equal(t, "0.0.0.0", opts.Host)
	assert.Equal(t, 9090, opts.Port)
	assert.Equal(t, "debug", opts.LogLevel)
}

func TestHandleSearch(t *testing.T) {
	t.Parallel()

	server := createTestServer(t)

	ctx := context.Background()
	req := mcp.CallToolRequest{}
	req.Params.Arguments = map[string]interface{}{
		"query": "authentication",
		"limit": float64(10),
	}

	result, err := server.handleSearch(ctx, req)
	require.NoError(t, err)
	require.NotNil(t, result)
}

func TestHandleSearchWithIndex(t *testing.T) {
	t.Parallel()

	server := createTestServer(t)

	ctx := context.Background()
	req := mcp.CallToolRequest{}
	req.Params.Arguments = map[string]interface{}{
		"query": "authentication",
		"index": "my-project",
		"limit": float64(5),
	}

	result, err := server.handleSearch(ctx, req)
	require.NoError(t, err)
	require.NotNil(t, result)
}

func TestHandleAsk(t *testing.T) {
	t.Parallel()

	server := createTestServer(t)

	ctx := context.Background()
	req := mcp.CallToolRequest{}
	req.Params.Arguments = map[string]interface{}{
		"question": "How does authentication work?",
	}

	result, err := server.handleAsk(ctx, req)
	require.NoError(t, err)
	require.NotNil(t, result)
}

func TestHandleAskWithIndex(t *testing.T) {
	t.Parallel()

	server := createTestServer(t)

	ctx := context.Background()
	req := mcp.CallToolRequest{}
	req.Params.Arguments = map[string]interface{}{
		"question": "How does authentication work?",
		"index":    "my-project",
	}

	result, err := server.handleAsk(ctx, req)
	require.NoError(t, err)
	require.NotNil(t, result)
}

func TestHandleListIndexes(t *testing.T) {
	t.Parallel()

	server := createTestServer(t)

	ctx := context.Background()
	req := mcp.CallToolRequest{}

	result, err := server.handleListIndexes(ctx, req)
	require.NoError(t, err)
	require.NotNil(t, result)
}

func TestHandleGetIndex(t *testing.T) {
	t.Parallel()

	server := createTestServer(t)

	ctx := context.Background()
	req := mcp.CallToolRequest{}
	req.Params.Arguments = map[string]interface{}{
		"name": "nonexistent",
	}

	result, err := server.handleGetIndex(ctx, req)
	require.NoError(t, err)
	require.NotNil(t, result)
}

func TestCreateResourceHandler(t *testing.T) {
	t.Parallel()

	server := createTestServer(t)

	handler := server.createResourceHandler("nonexistent")
	require.NotNil(t, handler)

	ctx := context.Background()
	req := mcp.ReadResourceRequest{}
	req.Params.URI = "pythia://index/nonexistent"

	_, err := handler(ctx, req)
	assert.Error(t, err)
}

func createTestServer(t *testing.T) *Server {
	t.Helper()

	tempDir := t.TempDir()
	server, err := NewServer(ServerOptions{
		DataDir:   tempDir,
		Transport: "stdio",
		LogLevel:  "info",
	})
	require.NoError(t, err)

	return server
}
