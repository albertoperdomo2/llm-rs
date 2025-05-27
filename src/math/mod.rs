pub mod matrix;
pub mod activations;

pub use matrix::{matmul, transpose};
pub use activations::{silu, softmax};