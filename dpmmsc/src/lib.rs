pub mod priors;
pub mod stats;
pub mod clusters;
pub mod model;
pub mod local;
pub mod global;

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        let result = 2 + 2;
        assert_eq!(result, 4);
    }
}
