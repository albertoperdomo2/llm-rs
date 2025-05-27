pub mod safetensors;
pub mod hf_hub;

pub use safetensors::SafeTensorsLoader;
pub use hf_hub::download_model;

use std::collections::HashMap;

// generic weight storage
pub type WeightTensor = Vec<f32>;
pub type ModelWeights = HashMap<String, WeightTensor>;

pub trait ModelLoader {
    fn load_weights(&self, path: &str) -> crate::Result<ModelWeights>;
}
