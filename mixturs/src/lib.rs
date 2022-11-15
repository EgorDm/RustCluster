extern crate core;

pub mod utils;
pub mod model;
pub mod metrics;
pub mod stats;
pub mod state;
pub mod params;
#[cfg(not(tarpaulin_include))]
pub mod callback;
#[cfg(not(tarpaulin_include))]
#[cfg(feature = "plot")]
pub mod plotting;

pub use model::Model;
pub use params::{FitOptions, ModelOptions};
pub use callback::MonitoringCallback;
pub use metrics::{NMI, AIC, BIC};
pub use stats::{NIW};

