pub mod utils;
pub mod clusters;
pub mod model;
pub mod options;
pub mod metrics;
pub mod plotting;
pub mod stats;
pub mod state;

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        let result = 2 + 2;
        assert_eq!(result, 4);
    }
}
