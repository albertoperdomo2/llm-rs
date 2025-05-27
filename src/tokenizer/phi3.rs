use std::collections::HashMap;
use serde_json::Value;

pub struct Phi3Tokenizer {
    vocab: HashMap<String, u32>,
    reverse_vocab: HashMap<u32, String>,
    vocab_size: usize,
}

impl Phi3Tokenizer {
    pub fn from_pretrained(model_dir: &str) -> crate::Result<Self> {
        let tokenizer_path = format!("{}/tokenizer.json", model_dir);
        Self::load_from_json(&tokenizer_path)
    }
    
    fn load_from_json(path: &str) -> crate::Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let json: Value = serde_json::from_str(&content)?;
        
        let model = &json["model"];
        let vocab = &model["vocab"];
        
        let mut token_to_id = HashMap::new();
        let mut id_to_token = HashMap::new();
        
        if let Some(vocab_obj) = vocab.as_object() {
            for (token, id) in vocab_obj {
                let id_val = id.as_u64().unwrap() as u32;
                token_to_id.insert(token.clone(), id_val);
                id_to_token.insert(id_val, token.clone());
            }
        }
        
        let vocab_size = token_to_id.len();
        
        Ok(Phi3Tokenizer {
            vocab: token_to_id,
            reverse_vocab: id_to_token,
            vocab_size,
        })
    }
    
    pub fn encode(&self, text: &str) -> Vec<u32> {
        // simple word-level encoding for now
        // a real implementation would handle BPE/SentencePiece
        // but that is TODO
        let mut tokens = Vec::new();
        
        for word in text.split_whitespace() {
            if let Some(&token_id) = self.vocab.get(word) {
                tokens.push(token_id);
            } else if let Some(&unk_id) = self.vocab.get("<unk>") {
                tokens.push(unk_id);
            } else if let Some(&unk_id) = self.vocab.get("[UNK]") {
                tokens.push(unk_id);
            } else {
                tokens.push(0);
            }
        }
        
        tokens
    }
    
    pub fn decode(&self, tokens: &[u32]) -> String {
        tokens
            .iter()
            .map(|&token_id| {
                self.reverse_vocab
                    .get(&token_id)
                    .unwrap_or(&"<unk>".to_string())
                    .clone()
            })
            .collect::<Vec<String>>()
            .join(" ")
    }
    
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }
}