//! Index storage and persistence
//!
//! Provides serialization and deserialization of indexes with
//! support for incremental updates (selective recomputation).

use std::fs::{self, File};
use std::io::{BufReader, BufWriter};
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::error::{CoreError, CoreResult};

/// Metadata for a stored index
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexMetadata {
    /// Version of the index format
    pub version: u32,
    /// Number of vectors in the index
    pub num_vectors: usize,
    /// Dimension of vectors
    pub dimension: usize,
    /// Timestamp of creation
    pub created_at: i64,
    /// Timestamp of last update
    pub updated_at: i64,
    /// Optional description
    pub description: Option<String>,
}

impl IndexMetadata {
    /// Current index format version
    pub const CURRENT_VERSION: u32 = 1;

    /// Create new metadata
    #[must_use]
    pub fn new(num_vectors: usize, dimension: usize) -> Self {
        let now = chrono::Utc::now().timestamp();
        Self {
            version: Self::CURRENT_VERSION,
            num_vectors,
            dimension,
            created_at: now,
            updated_at: now,
            description: None,
        }
    }
}

/// Storage backend trait for index persistence
pub trait StorageBackend: Send + Sync {
    /// Save index data to storage
    fn save(&self, path: &Path, data: &[u8]) -> CoreResult<()>;

    /// Load index data from storage
    fn load(&self, path: &Path) -> CoreResult<Vec<u8>>;

    /// Check if index exists
    fn exists(&self, path: &Path) -> bool;

    /// Delete index from storage
    fn delete(&self, path: &Path) -> CoreResult<()>;
}

/// Local filesystem storage backend
#[derive(Debug, Default)]
pub struct FileSystemStorage;

impl StorageBackend for FileSystemStorage {
    fn save(&self, path: &Path, data: &[u8]) -> CoreResult<()> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(path, data)?;
        Ok(())
    }

    fn load(&self, path: &Path) -> CoreResult<Vec<u8>> {
        Ok(fs::read(path)?)
    }

    fn exists(&self, path: &Path) -> bool {
        path.exists()
    }

    fn delete(&self, path: &Path) -> CoreResult<()> {
        if path.exists() {
            fs::remove_file(path)?;
        }
        Ok(())
    }
}

/// Index writer for serialization
pub struct IndexWriter<W: std::io::Write> {
    writer: BufWriter<W>,
}

impl IndexWriter<File> {
    /// Create a new index writer for a file
    pub fn create(path: impl AsRef<Path>) -> CoreResult<Self> {
        let path = path.as_ref();
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let file = File::create(path)?;
        Ok(Self {
            writer: BufWriter::new(file),
        })
    }
}

impl<W: std::io::Write> IndexWriter<W> {
    /// Write metadata
    pub fn write_metadata(&mut self, metadata: &IndexMetadata) -> CoreResult<()> {
        let json =
            serde_json::to_vec(metadata).map_err(|e| CoreError::Serialization(e.to_string()))?;
        self.write_chunk(b"META", &json)
    }

    /// Write a raw chunk
    fn write_chunk(&mut self, tag: &[u8; 4], data: &[u8]) -> CoreResult<()> {
        use std::io::Write;

        self.writer.write_all(tag)?;
        let len = (data.len() as u64).to_le_bytes();
        self.writer.write_all(&len)?;
        self.writer.write_all(data)?;
        Ok(())
    }
}

/// Index reader for deserialization
pub struct IndexReader<R: std::io::Read> {
    reader: BufReader<R>,
}

impl IndexReader<File> {
    /// Open an index file for reading
    pub fn open(path: impl AsRef<Path>) -> CoreResult<Self> {
        let file = File::open(path)?;
        Ok(Self {
            reader: BufReader::new(file),
        })
    }
}

impl<R: std::io::Read> IndexReader<R> {
    /// Read metadata
    pub fn read_metadata(&mut self) -> CoreResult<IndexMetadata> {
        let (tag, data) = self.read_chunk()?;
        if &tag != b"META" {
            return Err(CoreError::Deserialization("expected META chunk".into()));
        }
        serde_json::from_slice(&data).map_err(|e| CoreError::Deserialization(e.to_string()))
    }

    /// Read a raw chunk
    fn read_chunk(&mut self) -> CoreResult<([u8; 4], Vec<u8>)> {
        use std::io::Read;

        let mut tag = [0u8; 4];
        self.reader.read_exact(&mut tag)?;

        let mut len_bytes = [0u8; 8];
        self.reader.read_exact(&mut len_bytes)?;
        let len = u64::from_le_bytes(len_bytes) as usize;

        let mut data = vec![0u8; len];
        self.reader.read_exact(&mut data)?;

        Ok((tag, data))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;
    use tempfile::tempdir;

    #[test]
    fn test_filesystem_storage() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.bin");

        let storage = FileSystemStorage;
        let data = b"test data";

        storage.save(&path, data).unwrap();
        assert!(storage.exists(&path));

        let loaded = storage.load(&path).unwrap();
        assert_eq!(loaded, data);

        storage.delete(&path).unwrap();
        assert!(!storage.exists(&path));
    }

    #[test]
    fn test_filesystem_storage_nested_path() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("nested/deep/path/test.bin");

        let storage = FileSystemStorage;
        let data = b"nested data";

        storage.save(&path, data).unwrap();
        assert!(storage.exists(&path));

        let loaded = storage.load(&path).unwrap();
        assert_eq!(loaded, data);
    }

    #[test]
    fn test_filesystem_storage_delete_nonexistent() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("nonexistent.bin");

        let storage = FileSystemStorage;
        // Should not error when deleting non-existent file
        storage.delete(&path).unwrap();
    }

    #[test]
    fn test_filesystem_storage_load_nonexistent() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("nonexistent.bin");

        let storage = FileSystemStorage;
        assert!(storage.load(&path).is_err());
    }

    #[test]
    fn test_index_metadata_new() {
        let meta = IndexMetadata::new(100, 128);
        assert_eq!(meta.version, IndexMetadata::CURRENT_VERSION);
        assert_eq!(meta.num_vectors, 100);
        assert_eq!(meta.dimension, 128);
        assert!(meta.created_at > 0);
        assert_eq!(meta.created_at, meta.updated_at);
        assert!(meta.description.is_none());
    }

    #[test]
    fn test_index_metadata_serialization() {
        let mut meta = IndexMetadata::new(50, 64);
        meta.description = Some("test index".to_string());

        let json = serde_json::to_string(&meta).unwrap();
        let parsed: IndexMetadata = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.version, meta.version);
        assert_eq!(parsed.num_vectors, 50);
        assert_eq!(parsed.dimension, 64);
        assert_eq!(parsed.description, Some("test index".to_string()));
    }

    #[test]
    fn test_index_writer_create() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("index.leann");

        let _writer = IndexWriter::create(&path).unwrap();
        assert!(path.exists());
    }

    #[test]
    fn test_index_writer_create_nested() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("nested/dir/index.leann");

        let _writer = IndexWriter::create(&path).unwrap();
        assert!(path.exists());
    }

    #[test]
    fn test_index_writer_write_metadata() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("index.leann");

        let mut writer = IndexWriter::create(&path).unwrap();
        let meta = IndexMetadata::new(10, 32);
        writer.write_metadata(&meta).unwrap();
    }

    #[test]
    fn test_index_reader_open() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("index.leann");

        // Create a file with metadata first
        {
            let mut writer = IndexWriter::create(&path).unwrap();
            let meta = IndexMetadata::new(10, 32);
            writer.write_metadata(&meta).unwrap();
        }

        let _reader = IndexReader::open(&path).unwrap();
    }

    #[test]
    fn test_index_reader_read_metadata() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("index.leann");

        // Write metadata
        {
            let mut writer = IndexWriter::create(&path).unwrap();
            let mut meta = IndexMetadata::new(42, 128);
            meta.description = Some("test description".to_string());
            writer.write_metadata(&meta).unwrap();
        }

        // Read it back
        let mut reader = IndexReader::open(&path).unwrap();
        let meta = reader.read_metadata().unwrap();

        assert_eq!(meta.version, IndexMetadata::CURRENT_VERSION);
        assert_eq!(meta.num_vectors, 42);
        assert_eq!(meta.dimension, 128);
        assert_eq!(meta.description, Some("test description".to_string()));
    }

    #[test]
    fn test_index_writer_with_cursor() {
        let mut buffer = Vec::new();
        let cursor = Cursor::new(&mut buffer);
        let mut writer = IndexWriter {
            writer: BufWriter::new(cursor),
        };

        let meta = IndexMetadata::new(5, 16);
        writer.write_metadata(&meta).unwrap();
    }

    #[test]
    fn test_index_reader_invalid_tag() {
        // Create a file with invalid tag
        let dir = tempdir().unwrap();
        let path = dir.path().join("invalid.leann");

        {
            let mut file = File::create(&path).unwrap();
            use std::io::Write;
            file.write_all(b"BAAD").unwrap(); // Wrong tag
            file.write_all(&8u64.to_le_bytes()).unwrap();
            file.write_all(b"testdata").unwrap();
        }

        let mut reader = IndexReader::open(&path).unwrap();
        let result = reader.read_metadata();
        assert!(result.is_err());
    }

    #[test]
    fn test_index_roundtrip() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("roundtrip.leann");

        let original = IndexMetadata::new(1000, 256);

        // Write
        {
            let mut writer = IndexWriter::create(&path).unwrap();
            writer.write_metadata(&original).unwrap();
        }

        // Read
        {
            let mut reader = IndexReader::open(&path).unwrap();
            let loaded = reader.read_metadata().unwrap();

            assert_eq!(loaded.version, original.version);
            assert_eq!(loaded.num_vectors, original.num_vectors);
            assert_eq!(loaded.dimension, original.dimension);
            assert_eq!(loaded.created_at, original.created_at);
        }
    }
}
