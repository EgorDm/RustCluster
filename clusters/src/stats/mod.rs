pub mod priors;
mod covariance;
pub mod dp;
mod batch_mvn;

pub use covariance::*;
pub use priors::*;
pub use batch_mvn::*;
