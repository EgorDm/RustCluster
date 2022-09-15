extern crate core;

pub mod utils;
pub mod model;
pub mod metrics;
pub mod plotting;
pub mod stats;
pub mod state;
pub mod params;
pub mod callback;

pub use model::Model;
pub use params::{FitOptions, ModelOptions};
pub use callback::MonitoringCallback;
pub use metrics::{NMI, AIC, BIC};
pub use stats::{NIW};

