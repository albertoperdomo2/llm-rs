use crate::math::matmul;

pub fn token_embedding_lookup(
    token_ids: &[u32], 
    embedding_weights: &[f32], 
    vocab_size: usize, 
    hidden_size: usize
) -> Vec<f32> {
    let mut embeddings = Vec::new();
    
    for &token_id in token_ids {
        let start_idx = (token_id as usize) * hidden_size;
        let end_idx = start_idx + hidden_size;
        
        if start_idx < embedding_weights.len() {
            embeddings.extend_from_slice(&embedding_weights[start_idx..end_idx]);
        } else {
            // handle out-of-vocab tokens
            embeddings.extend(vec![0.0; hidden_size]);
        }
    }
    
    embeddings
}

pub fn language_model_head(
    hidden_states: &[f32],
    lm_head_weights: &[f32],
    seq_len: usize,
    hidden_size: usize,
    vocab_size: usize
) -> Vec<f32> {
    matmul(hidden_states, lm_head_weights, seq_len, hidden_size, vocab_size)
}
