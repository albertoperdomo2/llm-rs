use crate::models::phi3::Phi3Model;
use crate::tokenizer::simple::Tokenizer;
use crate::tokenizer::phi3::Phi3Tokenizer;

pub struct Generator {
    pub model: Phi3Model,
}

impl Generator {
    pub fn new(model: Phi3Model) -> Self {
        Self { model }
    }

    pub fn from_pretrained(model_dir: &str) -> crate::Result<Self> {
        let model = crate::models::phi3::Phi3Model::from_pretrained(model_dir)?;
        Ok(Generator {
            model,
        })
    }
    
    pub fn generate(&self, prompt: &str, max_tokens: usize) -> crate::Result<String> {
        // tokenize input
        let input_tokens = self.model.tokenizer.encode(prompt);
        println!("input tokens: {:?}", input_tokens);
        
        let mut output_tokens = input_tokens.clone();
        
        // generation loop
        for step in 0..max_tokens {
            // run forward pass
            let logits = self.forward_pass(&output_tokens)?;
            
            // sample next token
            let next_token = self.sample_next_token(&logits)?;
            
            output_tokens.push(next_token);
            
            // assuming 0 is EOS token
            if next_token == 0 {
                break;
            }
        }
        
        // decode output
        let generated_text = self.model.tokenizer.decode(&output_tokens);
        
        Ok(generated_text)
    }
    
    fn forward_pass(&self, tokens: &[u32]) -> crate::Result<Vec<f32>> {
        let seq_len = tokens.len();
        let hidden_size = self.model.config.hidden_size;
        let vocab_size = self.model.config.vocab_size;
        
        let embeddings = crate::layers::embedding::token_embedding_lookup(
            tokens,
            self.model.get_embeddings()?,
            vocab_size,
            hidden_size
        );
        
        // run through transformer layers
        let mut hidden_states = embeddings;
        
        for layer_idx in 0..self.model.num_layers.min(5) { // FIXME start with just 5 layers for testing
            let layer_weights = self.model.get_layer_for_your_architecture(layer_idx)?;
            
            hidden_states = crate::models::transformers::phi3_transformer_block(
                &hidden_states,
                &layer_weights.head_weights,
                &layer_weights.w_output,
                &layer_weights.w_gate,
                &layer_weights.w_up,
                &layer_weights.w_down,
                &layer_weights.attn_norm_weight,
                &layer_weights.ff_norm_weight,
                seq_len,
                hidden_size,
                self.model.config.intermediate_size,
                self.model.config.num_attention_heads
            );
        }
        
        let last_token_start = (seq_len - 1) * hidden_size;
        let last_token_hidden = &hidden_states[last_token_start..last_token_start + hidden_size];
        
        let logits = crate::layers::embedding::language_model_head(
            last_token_hidden,
            self.model.get_output_head()?,
            1, // just the last token
            hidden_size,
            vocab_size
        );
        
        Ok(logits)
    }
    
    fn sample_next_token(&self, logits: &[f32]) -> crate::Result<u32> {
        // simple greedy sampling: pick the token with highest probability
        let mut max_idx = 0;
        let mut max_val = logits[0];
        
        for (idx, &val) in logits.iter().enumerate() {
            if val > max_val {
                max_val = val;
                max_idx = idx;
            }
        }
        
        Ok(max_idx as u32)
    }
}
