use std::collections::HashMap;
use crate::loader::{ModelWeights, SafeTensorsLoader, ModelLoader};
use crate::layers::attention::HeadWeights;

pub struct Phi3Model {
    pub config: Phi3Config,
    pub weights: ModelWeights,
    pub num_layers: usize,
    pub tokenizer: crate::tokenizer::phi3::Phi3Tokenizer,
}

#[derive(Debug, Clone)]
pub struct Phi3Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
}

pub struct Phi3LayerWeights {
    pub qkv_proj: Vec<f32>,
    pub o_proj: Vec<f32>,
    pub gate_up_proj: Vec<f32>,
    pub down_proj: Vec<f32>,
    pub input_layernorm: Vec<f32>,
    pub post_attention_layernorm: Vec<f32>,
}

pub struct CustomArchitectureWeights {
    pub head_weights: Vec<HeadWeights>,
    pub w_output: Vec<f32>,
    pub w_gate: Vec<f32>,
    pub w_up: Vec<f32>,
    pub w_down: Vec<f32>,
    pub attn_norm_weight: Vec<f32>,
    pub ff_norm_weight: Vec<f32>,
}

impl Phi3Model {
    pub fn from_pretrained(model_dir: &str) -> crate::Result<Self> {
        let loader = SafeTensorsLoader;
        let weights = loader.load_weights(model_dir)?;

        let tokenizer = crate::tokenizer::phi3::Phi3Tokenizer::from_pretrained(model_dir)?;
        let config = Self::infer_config(&weights)?;
        let num_layers = Self::count_layers(&weights);

        Ok(Phi3Model {
            config,
            weights,
            num_layers,
            tokenizer,
        })
    }

    pub fn from_weights(weights: ModelWeights) -> crate::Result<Self> {
        let config = Self::infer_config(&weights)?;
        
        let num_layers = Self::count_layers(&weights);

        let tokenizer = crate::tokenizer::phi3::Phi3Tokenizer::from_pretrained("")?;
        
        println!("Phi-3 model config:");
        println!(" * vocabulary: {}", config.vocab_size);
        println!(" * hidden size: {}", config.hidden_size); 
        println!(" * intermediate: {}", config.intermediate_size);
        println!(" * layers: {}", num_layers);
        println!(" * attention heads: {}", config.num_attention_heads);

        Ok(Phi3Model {
            config,
            weights,
            num_layers,
            tokenizer,
        })
    }
    
    fn infer_config(weights: &ModelWeights) -> crate::Result<Phi3Config> {
        let embed_weight = weights.get("model.embed_tokens.weight")
            .ok_or("Missing embedding weights")?;
        
        // Embedding shape: [vocab_size, hidden_size]
        // We need to calculate these from the flat vector length
        // For now, use known Phi-3 mini values
        Ok(Phi3Config {
            vocab_size: 32064,
            hidden_size: 3072,
            intermediate_size: 8192,
            num_hidden_layers: 32,
            num_attention_heads: 32,
        })
    }
    
    fn count_layers(weights: &ModelWeights) -> usize {
        let mut max_layer = 0;
        for key in weights.keys() {
            if key.contains("model.layers.") {
                let parts: Vec<&str> = key.split('.').collect();
                if parts.len() > 2 {
                    if let Ok(layer_num) = parts[2].parse::<usize>() {
                        max_layer = max_layer.max(layer_num);
                    }
                }
            }
        }
        max_layer + 1
    }
    
    pub fn get_layer_weights(&self, layer_idx: usize) -> crate::Result<Phi3LayerWeights> {
        let prefix = format!("model.layers.{}", layer_idx);
        
        let qkv_weight = self.weights.get(&format!("{}.self_attn.qkv_proj.weight", prefix))
            .ok_or("Missing QKV weights")?;
        let o_proj_weight = self.weights.get(&format!("{}.self_attn.o_proj.weight", prefix))
            .ok_or("Missing output projection")?;
        let gate_up_weight = self.weights.get(&format!("{}.mlp.gate_up_proj.weight", prefix))
            .ok_or("Missing gate_up weights")?;
        let down_weight = self.weights.get(&format!("{}.mlp.down_proj.weight", prefix))
            .ok_or("Missing down projection")?;
        let input_norm = self.weights.get(&format!("{}.input_layernorm.weight", prefix))
            .ok_or("Missing input norm")?;
        let post_norm = self.weights.get(&format!("{}.post_attention_layernorm.weight", prefix))
            .ok_or("Missing post attention norm")?;
        
        Ok(Phi3LayerWeights {
            qkv_proj: qkv_weight.clone(),
            o_proj: o_proj_weight.clone(),
            gate_up_proj: gate_up_weight.clone(),
            down_proj: down_weight.clone(),
            input_layernorm: input_norm.clone(),
            post_attention_layernorm: post_norm.clone(),
        })
    }

    pub fn get_layer_for_your_architecture(&self, layer_idx: usize) -> crate::Result<CustomArchitectureWeights> {
        let phi3_weights = self.get_layer_weights(layer_idx)?;
        
        // split the combined matrices
        let (q_proj, k_proj, v_proj) = self.split_qkv_weights(&phi3_weights.qkv_proj);
        let (gate_proj, up_proj) = self.split_gate_up_weights(&phi3_weights.gate_up_proj);
        
        // convert to our HeadWeights format
        let head_weights = self.create_head_weights(q_proj, k_proj, v_proj)?;
        
        Ok(CustomArchitectureWeights {
            head_weights,
            w_output: phi3_weights.o_proj,
            w_gate: gate_proj,
            w_up: up_proj,
            w_down: phi3_weights.down_proj,
            attn_norm_weight: phi3_weights.input_layernorm,
            ff_norm_weight: phi3_weights.post_attention_layernorm,
        })
    }

    pub fn get_embeddings(&self) -> crate::Result<&Vec<f32>> {
        self.weights.get("model.embed_tokens.weight")
        .ok_or("missing token embeddings".into())
    }

    pub fn get_output_head(&self) -> crate::Result<&Vec<f32>> {
        self.get_embeddings()
    }

    pub fn split_qkv_weights(&self, qkv_proj: &[f32]) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        // QKV is [9216, 3072] where 9216 = 3 × 3072 (Q + K + V concatenated)
        let hidden_size = self.config.hidden_size;
        let qkv_size = hidden_size * hidden_size;
        
        // splitting into three equal parts
        let q_proj = qkv_proj[0..qkv_size].to_vec();
        let k_proj = qkv_proj[qkv_size..2*qkv_size].to_vec();
        let v_proj = qkv_proj[2*qkv_size..3*qkv_size].to_vec();
        
        (q_proj, k_proj, v_proj)
    }

    pub fn split_gate_up_weights(&self, gate_up_proj: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let hidden_size = self.config.hidden_size;
        let intermediate_size = self.config.intermediate_size;
        let gate_up_size = intermediate_size * hidden_size;
        
        // splitting into two equal parts
        let gate_proj = gate_up_proj[0..gate_up_size].to_vec();
        let up_proj = gate_up_proj[gate_up_size..2*gate_up_size].to_vec();
        
        (gate_proj, up_proj)
    }

    pub fn create_head_weights(&self, q: Vec<f32>, k: Vec<f32>, v: Vec<f32>) -> crate::Result<Vec<HeadWeights>> {
        let num_heads = self.config.num_attention_heads;
        let head_size = self.config.hidden_size / num_heads;
        let mut head_weights: Vec<HeadWeights> = Vec::new();
        
        for head in 0..num_heads {
            head_weights.push(
                HeadWeights {
                    w_q: q[(head * head_size) * self.config.hidden_size..((head + 1) * head_size) * self.config.hidden_size].to_vec(),
                    w_k: k[(head * head_size) * self.config.hidden_size..((head + 1) * head_size) * self.config.hidden_size].to_vec(),
                    w_v: v[(head * head_size) * self.config.hidden_size..((head + 1) * head_size) * self.config.hidden_size].to_vec(),
                }
            );
        }

        Ok(head_weights)
    }
}
