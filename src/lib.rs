//! Tiny LLM Inference Engine
//! 
//! A tiny, high-performance inference engine for small language models (1B-3B parameters)
//! with focus on Phi-3 architecture
#![feature(portable_simd)]

pub mod math;
pub mod layers;
pub mod models;
pub mod tokenizer;
pub mod loader;
pub mod generation;


// core result type
pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
