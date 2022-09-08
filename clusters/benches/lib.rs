#[macro_use]
extern crate criterion;

pub mod stats;
pub mod utils;
pub mod metrics;
pub mod dpm;


criterion_main!(
    stats::stats,
    utils::utils,
    metrics::metrics,
    dpm::dpm,
);