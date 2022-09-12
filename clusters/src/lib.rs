extern crate core;

pub mod utils;
pub mod model;
pub mod metrics;
pub mod plotting;
pub mod stats;
pub mod state;
pub mod params;

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        let result = 2 + 2;
        assert_eq!(result, 4);
    }
}
