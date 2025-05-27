pub mod simple;
pub mod phi3;

// pub use simple::Tokenizer;
pub use phi3::Phi3Tokenizer;

pub trait Tokenizer {
    fn encode(&self, text: &str) -> Vec<u32>;
    fn decode(&self, tokens: &[u32]) -> String;
    fn vocab_size(&self) -> usize;
}

impl Tokenizer for Phi3Tokenizer {
    fn encode(&self, text: &str) -> Vec<u32> {
        self.encode(text)
    }
    
    fn decode(&self, tokens: &[u32]) -> String {
        self.decode(tokens)
    }
    
    fn vocab_size(&self) -> usize {
        self.vocab_size()
    }
}
