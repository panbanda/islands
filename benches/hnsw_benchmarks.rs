//! HNSW graph benchmarks

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use islands::core::hnsw::{HnswConfig, HnswGraph};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use std::hint::black_box;

fn create_random_vectors(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    (0..n)
        .map(|_| (0..dim).map(|_| rng.r#gen::<f32>() * 2.0 - 1.0).collect())
        .collect()
}

fn benchmark_insert(c: &mut Criterion) {
    let mut group = c.benchmark_group("hnsw_insert");

    for n in [100, 500, 1000, 2000] {
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, &n| {
            let vectors = create_random_vectors(n, 128, 42);
            b.iter(|| {
                let mut graph = HnswGraph::new(HnswConfig::fast()).unwrap();
                for v in &vectors {
                    graph.insert(v.clone()).unwrap();
                }
                black_box(graph)
            });
        });
    }

    group.finish();
}

fn benchmark_search(c: &mut Criterion) {
    let mut group = c.benchmark_group("hnsw_search");

    for n in [1000, 5000, 10000] {
        let vectors = create_random_vectors(n, 128, 42);
        let mut graph = HnswGraph::new(HnswConfig::default()).unwrap();
        for v in &vectors {
            graph.insert(v.clone()).unwrap();
        }

        let query = vec![0.5f32; 128];

        group.throughput(Throughput::Elements(1));
        group.bench_with_input(BenchmarkId::new("k=10", n), &query, |b, query| {
            b.iter(|| black_box(graph.search(query, 10, 100).unwrap()));
        });

        group.bench_with_input(BenchmarkId::new("k=50", n), &query, |b, query| {
            b.iter(|| black_box(graph.search(query, 50, 200).unwrap()));
        });
    }

    group.finish();
}

fn benchmark_search_ef(c: &mut Criterion) {
    let mut group = c.benchmark_group("hnsw_search_ef");

    let n = 5000;
    let vectors = create_random_vectors(n, 128, 42);
    let mut graph = HnswGraph::new(HnswConfig::default()).unwrap();
    for v in &vectors {
        graph.insert(v.clone()).unwrap();
    }

    let query = vec![0.5f32; 128];

    for ef in [50, 100, 200, 400] {
        group.bench_with_input(BenchmarkId::from_parameter(ef), &ef, |b, &ef| {
            b.iter(|| black_box(graph.search(&query, 10, ef).unwrap()));
        });
    }

    group.finish();
}

fn benchmark_dimensions(c: &mut Criterion) {
    let mut group = c.benchmark_group("hnsw_dimensions");

    for dim in [32, 64, 128, 256, 512] {
        let n = 1000;
        let vectors = create_random_vectors(n, dim, 42);
        let mut graph = HnswGraph::new(HnswConfig::fast()).unwrap();
        for v in &vectors {
            graph.insert(v.clone()).unwrap();
        }

        let query: Vec<f32> = (0..dim).map(|_| 0.5).collect();

        group.bench_with_input(BenchmarkId::from_parameter(dim), &query, |b, query| {
            b.iter(|| black_box(graph.search(query, 10, 100).unwrap()));
        });
    }

    group.finish();
}

fn benchmark_serialization(c: &mut Criterion) {
    let mut group = c.benchmark_group("hnsw_serialization");

    for n in [500, 1000, 2000] {
        let vectors = create_random_vectors(n, 128, 42);
        let mut graph = HnswGraph::new(HnswConfig::fast()).unwrap();
        for v in &vectors {
            graph.insert(v.clone()).unwrap();
        }

        group.bench_with_input(BenchmarkId::new("serialize", n), &graph, |b, graph| {
            b.iter(|| black_box(graph.to_bytes().unwrap()));
        });

        let bytes = graph.to_bytes().unwrap();
        group.bench_with_input(BenchmarkId::new("deserialize", n), &bytes, |b, bytes| {
            b.iter(|| black_box(HnswGraph::from_bytes(bytes).unwrap()));
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    benchmark_insert,
    benchmark_search,
    benchmark_search_ef,
    benchmark_dimensions,
    benchmark_serialization,
);
criterion_main!(benches);
