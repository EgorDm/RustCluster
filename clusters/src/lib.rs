pub mod priors;
pub mod utils;
pub mod clusters;
pub mod model;
pub mod local;
pub mod global;
pub mod options;
pub mod metrics;
pub mod plotting;
pub mod stats;

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        let result = 2 + 2;
        assert_eq!(result, 4);
    }
}
