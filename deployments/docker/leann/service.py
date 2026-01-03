#!/usr/bin/env python3
"""
LEANN Service - HTTP wrapper for LEANN vector search.

Provides REST API endpoints for:
- Index building and management
- Semantic search
- Question answering with RAG
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, asdict
from flask import Flask, request, jsonify

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('leann-service')

app = Flask(__name__)

DATA_DIR = Path(os.environ.get('LEANN_DATA_DIR', '/data/leann'))
MODEL_NAME = os.environ.get('LEANN_MODEL', 'all-MiniLM-L6-v2')
DEVICE = os.environ.get('LEANN_DEVICE', 'cpu')

DATA_DIR.mkdir(parents=True, exist_ok=True)

indexes: dict = {}


@dataclass
class Document:
    id: str
    content: str
    metadata: dict


@dataclass
class SearchResult:
    id: str
    content: str
    score: float
    metadata: dict


def get_index_path(name: str) -> Path:
    return DATA_DIR / name


def load_index(name: str):
    """Load or create an index."""
    global indexes
    if name in indexes:
        return indexes[name]

    index_path = get_index_path(name)
    if not index_path.exists():
        return None

    try:
        from leann import LeannSearcher
        searcher = LeannSearcher(str(index_path))
        indexes[name] = {
            'searcher': searcher,
            'path': index_path
        }
        logger.info(f"Loaded index: {name}")
        return indexes[name]
    except Exception as e:
        logger.error(f"Failed to load index {name}: {e}")
        return None


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({'status': 'healthy', 'service': 'leann'})


@app.route('/indexes', methods=['GET'])
def list_indexes():
    """List all available indexes."""
    index_dirs = [d.name for d in DATA_DIR.iterdir() if d.is_dir()]
    return jsonify({'indexes': index_dirs})


@app.route('/indexes/<name>', methods=['GET'])
def get_index(name: str):
    """Get index information."""
    index_path = get_index_path(name)
    if not index_path.exists():
        return jsonify({'error': f'Index {name} not found'}), 404

    metadata_file = index_path / 'metadata.json'
    if metadata_file.exists():
        with open(metadata_file) as f:
            metadata = json.load(f)
    else:
        metadata = {'name': name}

    return jsonify(metadata)


@app.route('/indexes/<name>', methods=['DELETE'])
def delete_index(name: str):
    """Delete an index."""
    global indexes

    if name in indexes:
        del indexes[name]

    index_path = get_index_path(name)
    if index_path.exists():
        import shutil
        shutil.rmtree(index_path)
        return jsonify({'status': 'deleted', 'name': name})

    return jsonify({'error': f'Index {name} not found'}), 404


@app.route('/indexes/<name>/build', methods=['POST'])
def build_index(name: str):
    """Build an index from documents."""
    global indexes

    data = request.get_json()
    if not data:
        return jsonify({'error': 'Request body required'}), 400

    documents = data.get('documents', [])
    if not documents:
        return jsonify({'error': 'No documents provided'}), 400

    backend = data.get('backend', 'hnsw')
    force = data.get('force', False)

    index_path = get_index_path(name)

    if index_path.exists() and not force:
        return jsonify({'error': f'Index {name} already exists (use force=true to overwrite)'}), 409

    try:
        from leann import LeannBuilder

        builder = LeannBuilder(
            backend_name=backend,
            model_name=MODEL_NAME,
            device=DEVICE
        )

        for doc in documents:
            doc_id = doc.get('id', '')
            content = doc.get('content', '')
            metadata = doc.get('metadata', {})

            builder.add_text(content, metadata={'id': doc_id, **metadata})

        if index_path.exists():
            import shutil
            shutil.rmtree(index_path)

        builder.build_index(str(index_path))

        metadata = {
            'name': name,
            'document_count': len(documents),
            'backend': backend,
            'model': MODEL_NAME
        }
        with open(index_path / 'metadata.json', 'w') as f:
            json.dump(metadata, f)

        if name in indexes:
            del indexes[name]

        logger.info(f"Built index {name} with {len(documents)} documents")
        return jsonify({
            'status': 'built',
            'name': name,
            'documents': len(documents)
        })

    except ImportError as e:
        logger.error(f"LEANN not available: {e}")
        return jsonify({'error': 'LEANN library not available'}), 500
    except Exception as e:
        logger.error(f"Failed to build index {name}: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/indexes/<name>/search', methods=['POST'])
def search(name: str):
    """Search an index."""
    data = request.get_json()
    if not data:
        return jsonify({'error': 'Request body required'}), 400

    query = data.get('query', '')
    if not query:
        return jsonify({'error': 'Query required'}), 400

    top_k = data.get('limit', 10)
    threshold = data.get('threshold', 0.0)

    index = load_index(name)
    if not index:
        return jsonify({'error': f'Index {name} not found'}), 404

    try:
        searcher = index['searcher']
        results = searcher.search(query, top_k=top_k)

        response_results = []
        for r in results:
            score = getattr(r, 'score', 0.0)
            if score < threshold:
                continue

            response_results.append({
                'id': getattr(r, 'metadata', {}).get('id', ''),
                'content': getattr(r, 'text', ''),
                'score': score,
                'metadata': getattr(r, 'metadata', {})
            })

        return jsonify({'results': response_results})

    except Exception as e:
        logger.error(f"Search failed: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/indexes/<name>/ask', methods=['POST'])
def ask(name: str):
    """Answer a question using RAG."""
    data = request.get_json()
    if not data:
        return jsonify({'error': 'Request body required'}), 400

    question = data.get('question', '')
    if not question:
        return jsonify({'error': 'Question required'}), 400

    top_k = data.get('context_limit', 5)

    index = load_index(name)
    if not index:
        return jsonify({'error': f'Index {name} not found'}), 404

    try:
        searcher = index['searcher']
        results = searcher.search(question, top_k=top_k)

        context_parts = []
        sources = []
        for r in results:
            text = getattr(r, 'text', '')
            metadata = getattr(r, 'metadata', {})
            score = getattr(r, 'score', 0.0)

            context_parts.append(text)
            sources.append({
                'id': metadata.get('id', ''),
                'score': score,
                'file': metadata.get('file', '')
            })

        context = '\n\n---\n\n'.join(context_parts)

        return jsonify({
            'question': question,
            'context': context,
            'sources': sources
        })

    except Exception as e:
        logger.error(f"Ask failed: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/indexes/<name>/add', methods=['POST'])
def add_documents(name: str):
    """Add documents to an existing index (incremental)."""
    data = request.get_json()
    if not data:
        return jsonify({'error': 'Request body required'}), 400

    documents = data.get('documents', [])
    if not documents:
        return jsonify({'error': 'No documents provided'}), 400

    index_path = get_index_path(name)
    if not index_path.exists():
        return jsonify({'error': f'Index {name} not found. Use /build first.'}), 404

    return jsonify({
        'error': 'Incremental indexing not yet supported. Rebuild index with all documents.'
    }), 501


if __name__ == '__main__':
    port = int(os.environ.get('LEANN_PORT', 8081))
    debug = os.environ.get('LEANN_DEBUG', 'false').lower() == 'true'

    logger.info(f"Starting LEANN service on port {port}")
    logger.info(f"Data directory: {DATA_DIR}")
    logger.info(f"Model: {MODEL_NAME}")
    logger.info(f"Device: {DEVICE}")

    app.run(host='0.0.0.0', port=port, debug=debug)
