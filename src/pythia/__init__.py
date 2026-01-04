"""
Pythia - Codebase Indexing and Inquiry System

A semantic search system for codebases using LEANN with MCP interface support.
Designed for Kubernetes environments with EFS storage for multi-repository management.
"""

__version__ = "0.1.0"
__author__ = "Pythia Contributors"
__license__ = "MIT"

from pythia.indexer.service import IndexerService
from pythia.providers.base import GitProvider
from pythia.providers.github import GitHubProvider
from pythia.providers.gitlab import GitLabProvider
from pythia.providers.bitbucket import BitbucketProvider
from pythia.providers.gitea import GiteaProvider

__all__ = [
    "IndexerService",
    "GitProvider",
    "GitHubProvider",
    "GitLabProvider",
    "BitbucketProvider",
    "GiteaProvider",
]
