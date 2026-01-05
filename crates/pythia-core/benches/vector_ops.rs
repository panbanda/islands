//! Vector distance calculation benchmarks

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use pythia_core::distance::{Distance, DistanceMetric};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use std::hint::black_box;

fn create_random_vector(dim: usize, seed: u64) -> Vec<f32> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    (0..dim).map(|_| rng.r#gen::<f32>() * 2.0 - 1.0).collect()
}

fn benchmark_distance_metrics(c: &mut Criterion) {
    let mut group = c.benchmark_group("distance_metrics");

    let dim = 128;
    let a = create_random_vector(dim, 42);
    let b = create_random_vector(dim, 43);

    for metric in [
        DistanceMetric::Cosine,
        DistanceMetric::Euclidean,
        DistanceMetric::DotProduct,
        DistanceMetric::Manhattan,
    ] {
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{:?}", metric)),
            &(&a, &b),
            |bench, (a, b)| {
                bench.iter(|| black_box(metric.calculate(*a, *b).unwrap()));
            },
        );
    }

    group.finish();
}

fn benchmark_distance_dimensions(c: &mut Criterion) {
    let mut group = c.benchmark_group("distance_dimensions");

    for dim in [32, 64, 128, 256, 512, 1024] {
        let a = create_random_vector(dim, 42);
        let b = create_random_vector(dim, 43);
        let metric = DistanceMetric::Cosine;

        group.bench_with_input(
            BenchmarkId::from_parameter(dim),
            &(&a, &b),
            |bench, (a, b)| {
                bench.iter(|| black_box(metric.calculate(*a, *b).unwrap()));
            },
        );
    }

    group.finish();
}

fn benchmark_batch_distance(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_distance");

    let dim = 128;
    let query = create_random_vector(dim, 42);

    for n in [10, 50, 100, 500, 1000] {
        let vectors: Vec<Vec<f32>> = (0..n)
            .map(|i| create_random_vector(dim, i as u64))
            .collect();
        let vec_refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();
        let metric = DistanceMetric::Cosine;

        group.bench_with_input(BenchmarkId::from_parameter(n), &vec_refs, |bench, vecs| {
            bench.iter(|| black_box(metric.batch_calculate(&query, vecs).unwrap()));
        });
    }

    group.finish();
}

fn benchmark_squared_distance(c: &mut Criterion) {
    let mut group = c.benchmark_group("squared_distance");

    let dim = 128;
    let a = create_random_vector(dim, 42);
    let b = create_random_vector(dim, 43);
    let metric = DistanceMetric::Euclidean;

    group.bench_function("regular", |bench| {
        bench.iter(|| black_box(metric.calculate(&a, &b).unwrap()));
    });

    group.bench_function("squared", |bench| {
        bench.iter(|| black_box(metric.calculate_squared(&a, &b).unwrap()));
    });

    group.finish();
}

criterion_group!(
    benches,
    benchmark_distance_metrics,
    benchmark_distance_dimensions,
    benchmark_batch_distance,
    benchmark_squared_distance,
);
criterion_main!(benches);
