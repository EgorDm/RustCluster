use criterion::{criterion_group, Criterion};
use nalgebra::{DMatrix, DVector, Dynamic, OMatrix};
use rand::distributions::{Standard, Distribution, WeightedIndex};
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::SmallRng;
use mixturs::utils::{col_broadcast_add, replacement_sampling_weighted, reservoir_sampling_weighted, unique_with_indices};

fn bench_unique_with_indices(c: &mut Criterion) {
    let values: Vec<i32> = rand::thread_rng().sample_iter(Standard).take(10000).collect();
    c.bench_function("unique_with_indices", move |bh| bh.iter(|| unique_with_indices(&values, true)));
}

fn bench_sampling(c: &mut Criterion) {
    let data: DVector<f64> = DVector::new_random(64);
    let src = data.as_slice();

    let mut rng = SmallRng::from_seed([0; 32]);
    c.bench_function("reservoir_sampling_weighted_single", move |bh| bh.iter(|| {
        let mut dst = [0; 1];
        reservoir_sampling_weighted(&mut rng, src.iter().cloned(), &mut dst);
        dst
    }));

    let mut rng = SmallRng::from_seed([0; 32]);
    c.bench_function("reservoir_sampling_weighted_multiple", move |bh| bh.iter(|| {
        let mut dst = [0; 1];
        for _ in 0..1000 {
            reservoir_sampling_weighted(&mut rng, src.iter().cloned(), &mut dst);
        }
        dst
    }));

    let mut rng = SmallRng::from_seed([0; 32]);
    c.bench_function("statrs_weighted_sampling_single", move |bh| bh.iter(|| {
        let mut dst = [0; 1];
        replacement_sampling_weighted(&mut rng, src.iter().cloned(), &mut dst);
        dst
    }));

    let mut rng = SmallRng::from_seed([0; 32]);
    c.bench_function("statrs_weighted_sampling_multiple", move |bh| bh.iter(|| {
        let mut dst = [0; 1000];
        replacement_sampling_weighted(&mut rng, src.iter().cloned(), &mut dst);
        dst
    }));
}

fn bench_broadcast(c: &mut Criterion) {
    let mat = DMatrix::<f64>::new_random(100, 100);
    let vec = DVector::<f64>::new_random(100);

    c.bench_function("col_broadcast_add", |bh| bh.iter(|| col_broadcast_add(mat.clone(), &vec)));
}

criterion_group!(
    utils,
    bench_unique_with_indices,
    bench_sampling,
    bench_broadcast,
);