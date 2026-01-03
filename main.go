// Copyright 2025 Jon
// SPDX-License-Identifier: Apache-2.0

// Pythia is an AI-powered codebase analysis tool that uses LEANN for semantic
// search and provides an MCP interface for LLM integration.
package main

import (
	"os"

	"github.com/jon/pythia/cmd/pythia"
)

func main() {
	if err := pythia.Execute(); err != nil {
		os.Exit(1)
	}
}
