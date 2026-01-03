// Copyright 2025 Jon
// SPDX-License-Identifier: Apache-2.0

package git

import (
	"context"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestNewClient(t *testing.T) {
	t.Parallel()

	tempDir := t.TempDir()

	client, err := NewClient(ClientOptions{
		CacheDir: tempDir,
		Timeout:  5 * time.Minute,
	})
	require.NoError(t, err)
	require.NotNil(t, client)
}

func TestNewClientDefaultCacheDir(t *testing.T) {
	t.Parallel()

	client, err := NewClient(ClientOptions{})
	require.NoError(t, err)
	require.NotNil(t, client)

	assert.NotEmpty(t, client.cacheDir)
}

func TestNewClientDefaultTimeout(t *testing.T) {
	t.Parallel()

	tempDir := t.TempDir()

	client, err := NewClient(ClientOptions{
		CacheDir: tempDir,
	})
	require.NoError(t, err)

	assert.Equal(t, 10*time.Minute, client.timeout)
}

func TestNewClientWithProviders(t *testing.T) {
	t.Parallel()

	tempDir := t.TempDir()

	providers := []Provider{
		{Name: "github", Type: GitHub, Token: "token1"},
		{Name: "gitlab", Type: GitLab, Token: "token2"},
	}

	client, err := NewClient(ClientOptions{
		CacheDir:  tempDir,
		Providers: providers,
	})
	require.NoError(t, err)

	assert.Len(t, client.providers, 2)
}

func TestParseRepositoryURL(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name     string
		url      string
		expected *Repository
		wantErr  bool
	}{
		{
			name: "github https",
			url:  "https://github.com/user/repo",
			expected: &Repository{
				URL:      "https://github.com/user/repo",
				Name:     "repo",
				Owner:    "user",
				Provider: GitHub,
			},
		},
		{
			name: "github with .git suffix",
			url:  "https://github.com/user/repo.git",
			expected: &Repository{
				URL:      "https://github.com/user/repo.git",
				Name:     "repo",
				Owner:    "user",
				Provider: GitHub,
			},
		},
		{
			name: "github with branch",
			url:  "https://github.com/user/repo/tree/main",
			expected: &Repository{
				URL:      "https://github.com/user/repo/tree/main",
				Name:     "repo",
				Owner:    "user",
				Provider: GitHub,
				Branch:   "main",
			},
		},
		{
			name: "gitlab https",
			url:  "https://gitlab.com/group/project",
			expected: &Repository{
				URL:      "https://gitlab.com/group/project",
				Name:     "project",
				Owner:    "group",
				Provider: GitLab,
			},
		},
		{
			name: "bitbucket https",
			url:  "https://bitbucket.org/team/repo",
			expected: &Repository{
				URL:      "https://bitbucket.org/team/repo",
				Name:     "repo",
				Owner:    "team",
				Provider: Bitbucket,
			},
		},
		{
			name: "ssh url",
			url:  "git@github.com:user/repo.git",
			expected: &Repository{
				URL:      "https://github.com/user/repo.git",
				Name:     "repo",
				Owner:    "user",
				Provider: GitHub,
			},
		},
		{
			name: "unknown provider",
			url:  "https://custom-git.example.com/user/repo",
			expected: &Repository{
				URL:      "https://custom-git.example.com/user/repo",
				Name:     "repo",
				Owner:    "user",
				Provider: Generic,
			},
		},
		{
			name:    "invalid url missing path",
			url:     "https://github.com/user",
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			repo, err := ParseRepositoryURL(tt.url)

			if tt.wantErr {
				assert.Error(t, err)
				return
			}

			require.NoError(t, err)
			assert.Equal(t, tt.expected.Name, repo.Name)
			assert.Equal(t, tt.expected.Owner, repo.Owner)
			assert.Equal(t, tt.expected.Provider, repo.Provider)
			assert.Equal(t, tt.expected.Branch, repo.Branch)
		})
	}
}

func TestConvertSSHToHTTPS(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name     string
		ssh      string
		expected string
	}{
		{
			name:     "github ssh",
			ssh:      "git@github.com:user/repo.git",
			expected: "https://github.com/user/repo.git",
		},
		{
			name:     "gitlab ssh",
			ssh:      "git@gitlab.com:group/project.git",
			expected: "https://gitlab.com/group/project.git",
		},
		{
			name:     "bitbucket ssh",
			ssh:      "git@bitbucket.org:team/repo.git",
			expected: "https://bitbucket.org/team/repo.git",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			result := convertSSHToHTTPS(tt.ssh)
			assert.Equal(t, tt.expected, result)
		})
	}
}

func TestDetectProvider(t *testing.T) {
	t.Parallel()

	tests := []struct {
		host     string
		expected ProviderType
	}{
		{"github.com", GitHub},
		{"api.github.com", GitHub},
		{"gitlab.com", GitLab},
		{"gitlab.example.com", GitLab},
		{"bitbucket.org", Bitbucket},
		{"bitbucket.example.com", Bitbucket},
		{"custom-git.example.com", Generic},
		{"git.internal.corp", Generic},
	}

	for _, tt := range tests {
		t.Run(tt.host, func(t *testing.T) {
			t.Parallel()

			result := detectProvider(tt.host)
			assert.Equal(t, tt.expected, result)
		})
	}
}

func TestCloneInvalidURL(t *testing.T) {
	t.Parallel()

	tempDir := t.TempDir()

	client, err := NewClient(ClientOptions{
		CacheDir: tempDir,
	})
	require.NoError(t, err)

	ctx := context.Background()
	_, err = client.Clone(ctx, "not-a-valid-url")
	assert.Error(t, err)
}

func TestCloneContextCancellation(t *testing.T) {
	t.Parallel()

	tempDir := t.TempDir()

	client, err := NewClient(ClientOptions{
		CacheDir: tempDir,
		Timeout:  100 * time.Millisecond,
	})
	require.NoError(t, err)

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	_, err = client.Clone(ctx, "https://github.com/very-large/repository")
	assert.Error(t, err)
}

func TestGetAuthenticatedURL(t *testing.T) {
	t.Parallel()

	tempDir := t.TempDir()

	client, err := NewClient(ClientOptions{
		CacheDir: tempDir,
		Providers: []Provider{
			{Name: "github", Type: GitHub, Token: "secret-token"},
		},
	})
	require.NoError(t, err)

	repo := &Repository{
		URL:      "https://github.com/user/repo",
		Provider: GitHub,
	}

	authURL := client.getAuthenticatedURL(repo)
	assert.Contains(t, authURL, "secret-token")
}

func TestGetAuthenticatedURLNoToken(t *testing.T) {
	t.Parallel()

	tempDir := t.TempDir()

	client, err := NewClient(ClientOptions{
		CacheDir: tempDir,
	})
	require.NoError(t, err)

	repo := &Repository{
		URL:      "https://github.com/user/repo",
		Provider: GitHub,
	}

	authURL := client.getAuthenticatedURL(repo)
	assert.Equal(t, repo.URL, authURL)
}

func TestProviderType(t *testing.T) {
	t.Parallel()

	assert.Equal(t, ProviderType("github"), GitHub)
	assert.Equal(t, ProviderType("gitlab"), GitLab)
	assert.Equal(t, ProviderType("bitbucket"), Bitbucket)
	assert.Equal(t, ProviderType("generic"), Generic)
}

func TestRepository(t *testing.T) {
	t.Parallel()

	repo := Repository{
		URL:      "https://github.com/user/repo",
		Name:     "repo",
		Owner:    "user",
		Provider: GitHub,
		Branch:   "main",
	}

	assert.Equal(t, "https://github.com/user/repo", repo.URL)
	assert.Equal(t, "repo", repo.Name)
	assert.Equal(t, "user", repo.Owner)
	assert.Equal(t, GitHub, repo.Provider)
	assert.Equal(t, "main", repo.Branch)
}

func TestProvider(t *testing.T) {
	t.Parallel()

	provider := Provider{
		Name:    "my-github",
		Type:    GitHub,
		BaseURL: "https://github.com",
		Token:   "secret",
	}

	assert.Equal(t, "my-github", provider.Name)
	assert.Equal(t, GitHub, provider.Type)
	assert.Equal(t, "https://github.com", provider.BaseURL)
	assert.Equal(t, "secret", provider.Token)
}

func TestClientOptions(t *testing.T) {
	t.Parallel()

	opts := ClientOptions{
		CacheDir: "/cache",
		Timeout:  5 * time.Minute,
		Providers: []Provider{
			{Name: "test"},
		},
	}

	assert.Equal(t, "/cache", opts.CacheDir)
	assert.Equal(t, 5*time.Minute, opts.Timeout)
	assert.Len(t, opts.Providers, 1)
}

func TestCacheDirectoryCreation(t *testing.T) {
	t.Parallel()

	tempDir := t.TempDir()
	cacheDir := filepath.Join(tempDir, "nested", "cache", "dir")

	client, err := NewClient(ClientOptions{
		CacheDir: cacheDir,
	})
	require.NoError(t, err)
	require.NotNil(t, client)

	info, err := os.Stat(cacheDir)
	require.NoError(t, err)
	assert.True(t, info.IsDir())
}
