#[macro_use]
extern crate criterion;

pub mod stats;
pub mod utils;
pub mod metrics;


criterion_main!(
    stats::stats,
    utils::utils,
    metrics::metrics,
);