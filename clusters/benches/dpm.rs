use criterion::{criterion_group, Criterion};
use nalgebra::{DMatrix, RowDVector};
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use clusters::state::{LocalState, LocalWorker};
use clusters::stats::NIW;

fn bench_local_collect_stats(c: &mut Criterion) {
    let mut rng = StdRng::seed_from_u64(42);
    let data = DMatrix::from_fn(2, 1200, |_, _| rng.gen_range(0.0..1.0));
    let labels = RowDVector::from_fn(1200, |_, i| i / 300);
    let labels_aux = RowDVector::from_fn(1200, |_, i| i / 150 % 2);

    let local = LocalState::<NIW>::new(data.clone(), labels.clone(), labels_aux.clone());
    c.bench_function("collect_stats", move |bh| bh.iter(|| {
        let local = local.clone();
        local.collect_cluster_stats(4)
    }));
}

criterion_group!(
    dpm,
    bench_local_collect_stats,
);