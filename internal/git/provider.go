// Copyright 2025 Jon
// SPDX-License-Identifier: Apache-2.0

// Package git provides multi-provider Git repository support.
package git

import (
	"context"
	"errors"
	"fmt"
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"
)

var (
	// ErrUnsupportedProvider is returned for unknown Git providers.
	ErrUnsupportedProvider = errors.New("unsupported git provider")
	// ErrCloneFailed is returned when repository cloning fails.
	ErrCloneFailed = errors.New("clone failed")
)

// ProviderType represents a Git hosting provider.
type ProviderType string

const (
	// GitHub is github.com.
	GitHub ProviderType = "github"
	// GitLab is gitlab.com.
	GitLab ProviderType = "gitlab"
	// Bitbucket is bitbucket.org.
	Bitbucket ProviderType = "bitbucket"
	// Generic is any Git server.
	Generic ProviderType = "generic"
)

// Provider represents a Git hosting provider configuration.
type Provider struct {
	Name    string       `json:"name"`
	Type    ProviderType `json:"type"`
	BaseURL string       `json:"baseUrl"`
	Token   string       `json:"-"`
}

// Repository represents a Git repository.
type Repository struct {
	URL      string       `json:"url"`
	Name     string       `json:"name"`
	Owner    string       `json:"owner"`
	Provider ProviderType `json:"provider"`
	Branch   string       `json:"branch"`
}

// Client provides Git operations.
type Client struct {
	providers []Provider
	cacheDir  string
	timeout   time.Duration
}

// ClientOptions configures a Git client.
type ClientOptions struct {
	Providers []Provider
	CacheDir  string
	Timeout   time.Duration
}

// NewClient creates a new Git client.
func NewClient(opts ClientOptions) (*Client, error) {
	if opts.CacheDir == "" {
		home, err := os.UserHomeDir()
		if err != nil {
			return nil, fmt.Errorf("failed to get home directory: %w", err)
		}
		opts.CacheDir = filepath.Join(home, ".pythia", "git-cache")
	}

	if err := os.MkdirAll(opts.CacheDir, 0755); err != nil {
		return nil, fmt.Errorf("failed to create cache directory: %w", err)
	}

	if opts.Timeout == 0 {
		opts.Timeout = 10 * time.Minute
	}

	return &Client{
		providers: opts.Providers,
		cacheDir:  opts.CacheDir,
		timeout:   opts.Timeout,
	}, nil
}

// Clone clones a repository to the cache directory.
func (c *Client) Clone(ctx context.Context, repoURL string) (string, error) {
	repo, err := ParseRepositoryURL(repoURL)
	if err != nil {
		return "", err
	}

	destDir := filepath.Join(c.cacheDir, string(repo.Provider), repo.Owner, repo.Name)

	if info, err := os.Stat(destDir); err == nil && info.IsDir() {
		if err := c.pull(ctx, destDir); err != nil {
			os.RemoveAll(destDir)
		} else {
			return destDir, nil
		}
	}

	if err := os.MkdirAll(filepath.Dir(destDir), 0755); err != nil {
		return "", fmt.Errorf("failed to create directory: %w", err)
	}

	cloneURL := c.getAuthenticatedURL(repo)

	args := []string{"clone", "--depth", "1"}
	if repo.Branch != "" {
		args = append(args, "--branch", repo.Branch)
	}
	args = append(args, cloneURL, destDir)

	ctx, cancel := context.WithTimeout(ctx, c.timeout)
	defer cancel()

	cmd := exec.CommandContext(ctx, "git", args...)
	cmd.Env = append(os.Environ(), "GIT_TERMINAL_PROMPT=0")

	if output, err := cmd.CombinedOutput(); err != nil {
		return "", fmt.Errorf("%w: %s", ErrCloneFailed, string(output))
	}

	return destDir, nil
}

// ParseRepositoryURL parses a repository URL into its components.
func ParseRepositoryURL(rawURL string) (*Repository, error) {
	if strings.HasPrefix(rawURL, "git@") {
		rawURL = convertSSHToHTTPS(rawURL)
	}

	u, err := url.Parse(rawURL)
	if err != nil {
		return nil, fmt.Errorf("invalid repository URL: %w", err)
	}

	parts := strings.Split(strings.Trim(u.Path, "/"), "/")
	if len(parts) < 2 {
		return nil, fmt.Errorf("invalid repository path: %s", u.Path)
	}

	owner := parts[0]
	name := strings.TrimSuffix(parts[1], ".git")

	var branch string
	if len(parts) > 3 && parts[2] == "tree" {
		branch = parts[3]
	}

	provider := detectProvider(u.Host)

	return &Repository{
		URL:      rawURL,
		Name:     name,
		Owner:    owner,
		Provider: provider,
		Branch:   branch,
	}, nil
}

func (c *Client) pull(ctx context.Context, dir string) error {
	ctx, cancel := context.WithTimeout(ctx, c.timeout)
	defer cancel()

	cmd := exec.CommandContext(ctx, "git", "-C", dir, "pull", "--ff-only")
	cmd.Env = append(os.Environ(), "GIT_TERMINAL_PROMPT=0")
	return cmd.Run()
}

func (c *Client) getAuthenticatedURL(repo *Repository) string {
	for _, p := range c.providers {
		if p.Type == repo.Provider && p.Token != "" {
			u, err := url.Parse(repo.URL)
			if err != nil {
				return repo.URL
			}
			u.User = url.UserPassword("git", p.Token)
			return u.String()
		}
	}
	return repo.URL
}

func convertSSHToHTTPS(sshURL string) string {
	sshURL = strings.TrimPrefix(sshURL, "git@")
	parts := strings.SplitN(sshURL, ":", 2)
	if len(parts) != 2 {
		return sshURL
	}
	return "https://" + parts[0] + "/" + parts[1]
}

func detectProvider(host string) ProviderType {
	host = strings.ToLower(host)
	switch {
	case strings.Contains(host, "github"):
		return GitHub
	case strings.Contains(host, "gitlab"):
		return GitLab
	case strings.Contains(host, "bitbucket"):
		return Bitbucket
	default:
		return Generic
	}
}
