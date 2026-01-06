//! Product Quantization benchmarks

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use islands::core::pq::{PQConfig, ProductQuantizer};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use std::hint::black_box;

fn create_random_vectors(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    (0..n)
        .map(|_| (0..dim).map(|_| rng.r#gen::<f32>() * 2.0 - 1.0).collect())
        .collect()
}

fn benchmark_training(c: &mut Criterion) {
    let mut group = c.benchmark_group("pq_training");

    for n in [100, 500, 1000] {
        let vectors = create_random_vectors(n, 128, 42);

        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &vectors, |b, vectors| {
            b.iter(|| {
                let mut pq = ProductQuantizer::new(
                    128,
                    PQConfig {
                        num_subquantizers: 8,
                        num_centroids: 64,
                        training_iterations: 10,
                        seed: Some(42),
                    },
                )
                .unwrap();
                pq.train(vectors).unwrap();
                black_box(pq)
            });
        });
    }

    group.finish();
}

fn benchmark_encoding(c: &mut Criterion) {
    let mut group = c.benchmark_group("pq_encoding");

    let dim = 128;
    let vectors = create_random_vectors(1000, dim, 42);

    let mut pq = ProductQuantizer::new(
        dim,
        PQConfig {
            num_subquantizers: 8,
            num_centroids: 256,
            training_iterations: 15,
            seed: Some(42),
        },
    )
    .unwrap();
    pq.train(&vectors).unwrap();

    let query = &vectors[0];

    group.throughput(Throughput::Elements(1));
    group.bench_function("single", |b| {
        b.iter(|| black_box(pq.encode(query).unwrap()));
    });

    group.throughput(Throughput::Elements(100));
    group.bench_function("batch_100", |b| {
        b.iter(|| {
            let codes: Vec<_> = vectors[..100]
                .iter()
                .map(|v| pq.encode(v).unwrap())
                .collect();
            black_box(codes)
        });
    });

    group.finish();
}

fn benchmark_decoding(c: &mut Criterion) {
    let mut group = c.benchmark_group("pq_decoding");

    let dim = 128;
    let vectors = create_random_vectors(500, dim, 42);

    let mut pq = ProductQuantizer::new(
        dim,
        PQConfig {
            num_subquantizers: 8,
            num_centroids: 256,
            training_iterations: 15,
            seed: Some(42),
        },
    )
    .unwrap();
    pq.train(&vectors).unwrap();

    let codes: Vec<_> = vectors.iter().map(|v| pq.encode(v).unwrap()).collect();

    group.throughput(Throughput::Elements(1));
    group.bench_function("single", |b| {
        b.iter(|| black_box(pq.decode(&codes[0]).unwrap()));
    });

    group.throughput(Throughput::Elements(100));
    group.bench_function("batch_100", |b| {
        b.iter(|| {
            let decoded: Vec<_> = codes[..100].iter().map(|c| pq.decode(c).unwrap()).collect();
            black_box(decoded)
        });
    });

    group.finish();
}

fn benchmark_asymmetric_distance(c: &mut Criterion) {
    let mut group = c.benchmark_group("pq_asymmetric_distance");

    let dim = 128;
    let vectors = create_random_vectors(500, dim, 42);

    let mut pq = ProductQuantizer::new(
        dim,
        PQConfig {
            num_subquantizers: 8,
            num_centroids: 256,
            training_iterations: 15,
            seed: Some(42),
        },
    )
    .unwrap();
    pq.train(&vectors).unwrap();

    let query = &vectors[0];
    let codes: Vec<_> = vectors[1..101]
        .iter()
        .map(|v| pq.encode(v).unwrap())
        .collect();

    group.bench_function("direct", |b| {
        b.iter(|| {
            let distances: Vec<_> = codes
                .iter()
                .map(|c| pq.asymmetric_distance(query, c).unwrap())
                .collect();
            black_box(distances)
        });
    });

    group.finish();
}

fn benchmark_table_distance(c: &mut Criterion) {
    let mut group = c.benchmark_group("pq_table_distance");

    let dim = 128;
    let vectors = create_random_vectors(500, dim, 42);

    let mut pq = ProductQuantizer::new(
        dim,
        PQConfig {
            num_subquantizers: 8,
            num_centroids: 256,
            training_iterations: 15,
            seed: Some(42),
        },
    )
    .unwrap();
    pq.train(&vectors).unwrap();

    let query = &vectors[0];
    let codes: Vec<_> = vectors[1..101]
        .iter()
        .map(|v| pq.encode(v).unwrap())
        .collect();
    let tables = pq.build_distance_tables(query).unwrap();

    group.bench_function("table_lookup", |b| {
        b.iter(|| {
            let distances: Vec<_> = codes
                .iter()
                .map(|c| pq.table_distance(&tables, c))
                .collect();
            black_box(distances)
        });
    });

    group.bench_function("build_tables", |b| {
        b.iter(|| black_box(pq.build_distance_tables(query).unwrap()));
    });

    group.finish();
}

fn benchmark_subquantizer_counts(c: &mut Criterion) {
    let mut group = c.benchmark_group("pq_subquantizers");

    let dim = 128;
    let vectors = create_random_vectors(500, dim, 42);

    for sq in [4, 8, 16, 32] {
        let mut pq = ProductQuantizer::new(
            dim,
            PQConfig {
                num_subquantizers: sq,
                num_centroids: 256,
                training_iterations: 10,
                seed: Some(42),
            },
        )
        .unwrap();
        pq.train(&vectors).unwrap();

        let query = &vectors[0];

        group.bench_with_input(BenchmarkId::from_parameter(sq), &query, |b, query| {
            b.iter(|| black_box(pq.encode(query).unwrap()));
        });
    }

    group.finish();
}

fn benchmark_serialization(c: &mut Criterion) {
    let mut group = c.benchmark_group("pq_serialization");

    let dim = 128;
    let vectors = create_random_vectors(500, dim, 42);

    for sq in [8, 16] {
        let mut pq = ProductQuantizer::new(
            dim,
            PQConfig {
                num_subquantizers: sq,
                num_centroids: 256,
                training_iterations: 10,
                seed: Some(42),
            },
        )
        .unwrap();
        pq.train(&vectors).unwrap();

        group.bench_with_input(BenchmarkId::new("serialize", sq), &pq, |b, pq| {
            b.iter(|| black_box(pq.to_bytes().unwrap()));
        });

        let bytes = pq.to_bytes().unwrap();
        group.bench_with_input(BenchmarkId::new("deserialize", sq), &bytes, |b, bytes| {
            b.iter(|| black_box(ProductQuantizer::from_bytes(bytes).unwrap()));
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    benchmark_training,
    benchmark_encoding,
    benchmark_decoding,
    benchmark_asymmetric_distance,
    benchmark_table_distance,
    benchmark_subquantizer_counts,
    benchmark_serialization,
);
criterion_main!(benches);
