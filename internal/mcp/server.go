// Copyright 2025 Jon
// SPDX-License-Identifier: Apache-2.0

// Package mcp provides an MCP server for LLM integration.
package mcp

import (
	"context"
	"fmt"

	"github.com/jon/pythia/internal/analyzer"
	"github.com/mark3labs/mcp-go/mcp"
	"github.com/mark3labs/mcp-go/server"
)

// ServerOptions configures the MCP server.
type ServerOptions struct {
	DataDir   string
	Transport string
	Host      string
	Port      int
	LogLevel  string
}

// Server is a Pythia MCP server.
type Server struct {
	opts     ServerOptions
	analyzer *analyzer.Analyzer
	mcp      *server.MCPServer
}

// NewServer creates a new MCP server.
func NewServer(opts ServerOptions) (*Server, error) {
	a, err := analyzer.New(analyzer.Options{
		DataDir:  opts.DataDir,
		LogLevel: opts.LogLevel,
	})
	if err != nil {
		return nil, fmt.Errorf("failed to create analyzer: %w", err)
	}

	mcpServer := server.NewMCPServer(
		"pythia",
		"1.0.0",
		server.WithResourceCapabilities(true, true),
		server.WithToolCapabilities(true),
	)

	s := &Server{
		opts:     opts,
		analyzer: a,
		mcp:      mcpServer,
	}

	s.registerTools()
	s.registerResources()

	return s, nil
}

// Serve starts the MCP server.
func (s *Server) Serve(ctx context.Context) error {
	if s.opts.Transport == "http" {
		return s.serveHTTP(ctx)
	}
	return s.serveStdio(ctx)
}

func (s *Server) serveStdio(ctx context.Context) error {
	return server.ServeStdio(s.mcp)
}

func (s *Server) serveHTTP(ctx context.Context) error {
	addr := fmt.Sprintf("%s:%d", s.opts.Host, s.opts.Port)
	return server.NewSSEServer(s.mcp).Start(addr)
}

func (s *Server) registerTools() {
	s.mcp.AddTool(mcp.NewTool("search",
		mcp.WithDescription("Search indexed codebases using semantic search"),
		mcp.WithString("query",
			mcp.Required(),
			mcp.Description("Natural language search query"),
		),
		mcp.WithString("index",
			mcp.Description("Specific index to search (optional)"),
		),
		mcp.WithNumber("limit",
			mcp.Description("Maximum results to return (default: 10)"),
		),
	), s.handleSearch)

	s.mcp.AddTool(mcp.NewTool("ask",
		mcp.WithDescription("Ask a question about the codebase"),
		mcp.WithString("question",
			mcp.Required(),
			mcp.Description("Question about the code"),
		),
		mcp.WithString("index",
			mcp.Description("Specific index to query (optional)"),
		),
	), s.handleAsk)

	s.mcp.AddTool(mcp.NewTool("list_indexes",
		mcp.WithDescription("List all indexed codebases"),
	), s.handleListIndexes)

	s.mcp.AddTool(mcp.NewTool("get_index",
		mcp.WithDescription("Get information about a specific index"),
		mcp.WithString("name",
			mcp.Required(),
			mcp.Description("Index name"),
		),
	), s.handleGetIndex)
}

func (s *Server) registerResources() {
	indexes, _ := s.analyzer.GetIndexes()
	for _, idx := range indexes {
		s.mcp.AddResource(mcp.NewResource(
			fmt.Sprintf("pythia://index/%s", idx.Name),
			idx.Name,
			mcp.WithResourceDescription(fmt.Sprintf("Indexed codebase: %s (%d files)", idx.Path, idx.FileCount)),
			mcp.WithMIMEType("application/json"),
		), s.createResourceHandler(idx.Name))
	}
}

func (s *Server) createResourceHandler(indexName string) server.ResourceHandlerFunc {
	return func(ctx context.Context, req mcp.ReadResourceRequest) ([]mcp.ResourceContents, error) {
		idx, err := s.analyzer.GetIndex(indexName)
		if err != nil {
			return nil, err
		}
		content := fmt.Sprintf(`{"name":"%s","path":"%s","files":%d}`, idx.Name, idx.Path, idx.FileCount)
		return []mcp.ResourceContents{
			mcp.TextResourceContents{
				URI:      req.Params.URI,
				MIMEType: "application/json",
				Text:     content,
			},
		}, nil
	}
}

func (s *Server) handleSearch(ctx context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
	query, _ := req.Params.Arguments["query"].(string)
	index, _ := req.Params.Arguments["index"].(string)
	limit := 10
	if l, ok := req.Params.Arguments["limit"].(float64); ok {
		limit = int(l)
	}

	results, err := s.analyzer.Search(ctx, analyzer.SearchOptions{
		Query: query,
		Index: index,
		Limit: limit,
	})
	if err != nil {
		return mcp.NewToolResultError(err.Error()), nil
	}

	var text string
	for i, r := range results {
		text += fmt.Sprintf("[%d] %s (score: %.2f)\n%s\n\n", i+1, r.File, r.Score, r.Preview)
	}

	if text == "" {
		text = "No results found."
	}

	return mcp.NewToolResultText(text), nil
}

func (s *Server) handleAsk(ctx context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
	question, _ := req.Params.Arguments["question"].(string)
	index, _ := req.Params.Arguments["index"].(string)

	answer, err := s.analyzer.Ask(ctx, question, index)
	if err != nil {
		return mcp.NewToolResultError(err.Error()), nil
	}

	return mcp.NewToolResultText(answer), nil
}

func (s *Server) handleListIndexes(ctx context.Context, _ mcp.CallToolRequest) (*mcp.CallToolResult, error) {
	indexes, err := s.analyzer.GetIndexes()
	if err != nil {
		return mcp.NewToolResultError(err.Error()), nil
	}

	if len(indexes) == 0 {
		return mcp.NewToolResultText("No indexes found."), nil
	}

	var text string
	for _, idx := range indexes {
		text += fmt.Sprintf("- %s: %s (%d files)\n", idx.Name, idx.Path, idx.FileCount)
	}

	return mcp.NewToolResultText(text), nil
}

func (s *Server) handleGetIndex(ctx context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
	name, _ := req.Params.Arguments["name"].(string)

	idx, err := s.analyzer.GetIndex(name)
	if err != nil {
		return mcp.NewToolResultError(err.Error()), nil
	}

	text := fmt.Sprintf("Index: %s\nPath: %s\nFiles: %d\nSize: %d bytes\nUpdated: %s",
		idx.Name, idx.Path, idx.FileCount, idx.Size, idx.UpdatedAt.Format("2006-01-02 15:04:05"))

	return mcp.NewToolResultText(text), nil
}
