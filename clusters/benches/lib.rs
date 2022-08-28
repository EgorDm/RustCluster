#[macro_use]
extern crate criterion;

pub mod stats;


criterion_main!(
    stats::stats,
);