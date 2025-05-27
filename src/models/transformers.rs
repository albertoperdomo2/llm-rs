use crate::layers::{complete_multi_head_attention, swiglu_feedforward, rms_norm};
use crate::layers::HeadWeights;

pub fn phi3_transformer_block(
    input: &[f32],
    head_weights: &[HeadWeights],
    w_output: &[f32],
    w_gate: &[f32],
    w_up: &[f32], 
    w_down: &[f32],
    attn_norm_weight: &[f32],
    ff_norm_weight: &[f32],
    seq_len: usize,
    hidden_size: usize,
    intermediate_size: usize,
    num_heads: usize
) -> Vec<f32> {
    let normalized_input = rms_norm(input, attn_norm_weight, seq_len, hidden_size);
    
    let attention_out = complete_multi_head_attention(
        &normalized_input, num_heads, seq_len, hidden_size, head_weights, w_output
    );
    
    // residual connection (add input back)
    let mut after_attention = Vec::new();
    for (i, (&att, &inp)) in attention_out.iter().zip(input.iter()).enumerate() {
        after_attention.push(att + inp);
    }
    
    // pre-feed-forward normalization  
    let normalized_attention = rms_norm(&after_attention, ff_norm_weight, seq_len, hidden_size);
    
    // SwiGLU feed-forward
    let ff_out = swiglu_feedforward(
        &normalized_attention, w_gate, w_up, w_down, 
        seq_len, hidden_size, intermediate_size
    );
    
    // final residual connection
    let mut output = Vec::new();
    for (i, (&ff, &att)) in ff_out.iter().zip(after_attention.iter()).enumerate() {
        output.push(ff + att);
    }
    
    output
}
