use std::collections::HashMap;
use std::fs::File;
use std::path::Path;
use serde_json::Value;
use std::io::{Read, Seek, SeekFrom};

use crate::loader::{ModelWeights, WeightTensor, ModelLoader};

pub struct SafeTensorsLoader;

impl ModelLoader for SafeTensorsLoader {
    fn load_weights(&self, model_dir: &str) -> crate::Result<ModelWeights> {
        // check wether it is a directory with index file or single file
        let path = Path::new(model_dir);
        
        if path.is_dir() {
            self.load_sharded_model(model_dir)
        } else {
            self.load_single_file(model_dir)
        }
    }
}

impl SafeTensorsLoader {
    fn load_sharded_model(&self, model_dir: &str) -> crate::Result<ModelWeights> {
        let index_path = format!("{}/model.safetensors.index.json", model_dir);
        
        println!("loading sharded model from: {}", model_dir);
        let index_content = std::fs::read_to_string(&index_path)?;
        
        let first_shard = format!("{}/model-00001-of-00002.safetensors", model_dir);
        let second_shard = format!("{}/model-00002-of-00002.safetensors", model_dir);
        
        let mut shard_one = self.load_single_file(&first_shard)?;
        let mut shard_two = self.load_single_file(&second_shard)?;

        for (name, tensor) in shard_two {
            shard_one.insert(name, tensor);
        }

        Ok(shard_one)
    }

    fn load_single_file(&self, path: &str) -> crate::Result<ModelWeights> {
        println!("loading SafeTensors file: {}", path);
        
        let mut file = File::open(path)?;
        
        // read header
        let mut header_len_bytes = [0u8; 8];
        file.read_exact(&mut header_len_bytes)?;
        let header_len = u64::from_le_bytes(header_len_bytes) as usize;
        
        let mut header_bytes = vec![0u8; header_len];
        file.read_exact(&mut header_bytes)?;
        let header_json = String::from_utf8(header_bytes)?;
        
        // parse JSON metadata
        let metadata: Value = serde_json::from_str(&header_json)?;
        
        let mut weights = HashMap::new();
        let data_start_offset = 8 + header_len; // start of tensor data
        
        // parse each tensor
        for (name, info) in metadata.as_object().unwrap() {
            if name.starts_with("__") {
                continue; // skip metadata
            }
            
            let tensor_info = info.as_object().unwrap();
            let dtype = tensor_info["dtype"].as_str().unwrap();
            let shape: Vec<usize> = tensor_info["shape"]
                .as_array().unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as usize)
                .collect();
            let offsets: Vec<usize> = tensor_info["data_offsets"]
                .as_array().unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as usize)
                .collect();
            
            // println!("loading tensor: {} {:?} {}", name, shape, dtype);
            
            // read tensor data
            if dtype == "BF16" {
                let tensor_data = self.read_bf16_tensor(&mut file, data_start_offset + offsets[0], offsets[1] - offsets[0])?;
                weights.insert(name.clone(), tensor_data);
            } else {
                println!("skipping unsupported dtype: {}", dtype);
            }
        }
        
        println!("loaded {} tensors", weights.len());
        Ok(weights)
    }
    
    fn read_bf16_tensor(&self, file: &mut File, offset: usize, size_bytes: usize) -> crate::Result<Vec<f32>> {
        // seek to tensor location
        file.seek(SeekFrom::Start(offset as u64))?;
        
        // read raw bytes
        let mut buffer = vec![0u8; size_bytes];
        file.read_exact(&mut buffer)?;
        
        // convert BF16 to F32
        let num_elements = size_bytes / 2; // BF16 = 2 bytes per element
        let mut tensor = Vec::with_capacity(num_elements);
        
        for i in 0..num_elements {
            let bf16_bytes = [buffer[i * 2], buffer[i * 2 + 1]];
            let f32_value = self.bf16_to_f32(bf16_bytes);
            tensor.push(f32_value);
        }
        
        Ok(tensor)
    }
    
    fn bf16_to_f32(&self, bf16_bytes: [u8; 2]) -> f32 {
        // the BF16 to F32 conversion
        // BF16: 1 sign bit + 8 exponent bits + 7 mantissa bits
        // F32: 1 sign bit + 8 exponent bits + 23 mantissa bits
        
        let bf16_value = u16::from_le_bytes(bf16_bytes);
        let f32_bits = (bf16_value as u32) << 16; // shift BF16 to upper 16 bits of F32
        
        f32::from_bits(f32_bits)
    }
}

#[derive(Debug)]
struct TensorInfo {
    dtype: String,
    shape: Vec<usize>,
    data_offsets: (usize, usize), // (start, end)
}

fn read_safetensors_header(file: &mut File) -> crate::Result<String> {
    // SafeTensors format: 8-byte header length + JSON header + tensor data
    let mut header_len_bytes = [0u8; 8];
    file.read_exact(&mut header_len_bytes)?;

    let header_len = u64::from_le_bytes(header_len_bytes) as usize;

    let mut header_bytes = vec![0u8; header_len];
    file.read_exact(&mut header_bytes)?;

    Ok(String::from_utf8(header_bytes)?)
}

fn parse_tensor_metadata(header_json: &str) -> crate::Result<HashMap<String, TensorInfo>> {
    todo!("implement");
}

fn read_tensor_data(file: &mut File, info: &TensorInfo) -> crate::Result<WeightTensor> {
    todo!("implement");
}
