use remoc::prelude::*;
use std::time::Duration;
use crate::global::GlobalState;
use crate::local::LocalStats;
use crate::stats::{NIW, NIWStats};

/// TCP port the server is listening on.
pub const TCP_PORT: u16 = 9871;

/// Increasing the counter failed.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub enum IncreaseError {
    /// An overflow would occur.
    Overflow {
        /// The current value of the counter.
        current_value: u32,
    },
    /// The RTC call failed.
    Call(rtc::CallError),
}

impl From<rtc::CallError> for IncreaseError {
    fn from(err: rtc::CallError) -> Self {
        Self::Call(err)
    }
}

/// Remote counting service.
#[rtc::remote]
pub trait Counter {
    async fn collect_data_stats(&self) -> Result<NIWStats, rtc::CallError>;

    async fn collect_stats(&self, n_clusters: usize) -> Result<LocalStats<NIW>, rtc::CallError>;

    async fn update_sample_labels(&mut self, global_state: GlobalState<NIW>, is_final: bool) -> Result<(), rtc::CallError>;

    async fn update_reset_clusters(&mut self, bad_clusters: Vec<usize>) -> Result<(), rtc::CallError>;

    async fn update_remove_clusters(&mut self, removed_idx: Vec<usize>) -> Result<(), rtc::CallError>;

    async fn apply_split(&mut self, split_idx: Vec<(usize, usize)>) -> Result<(), rtc::CallError>;

    async fn apply_merge(&mut self, merge_idx: Vec<(usize, usize)>) -> Result<(), rtc::CallError>;

    /// Obtain the current value of the counter.
    async fn value(&self) -> Result<u32, rtc::CallError>;

    /// Watch the current value of the counter for immediate notification
    /// when it changes.
    async fn watch(&mut self) -> Result<rch::watch::Receiver<u32>, rtc::CallError>;

    /// Increase the counter's value by the provided number.
    async fn increase(&mut self, by: u32) -> Result<(), IncreaseError>;

    /// Counts to the current value of the counter with the specified
    /// delay between each step.
    async fn count_to_value(
        &self, step: u32, delay: Duration,
    ) -> Result<rch::mpsc::Receiver<u32>, rtc::CallError>;
}