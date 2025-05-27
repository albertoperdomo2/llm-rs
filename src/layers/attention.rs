use crate::math::{matmul, transpose, softmax};

#[derive(Debug, Clone)]
pub struct HeadWeights {
    pub w_q: Vec<f32>,
    pub w_k: Vec<f32>,
    pub w_v: Vec<f32>,
}

pub fn simple_attention(
    input: &[f32],
    w_q: &[f32],
    w_k: &[f32],
    w_v: &[f32],
    seq_len: usize,
    hidden_size: usize,
) -> Vec<f32> {
    let q = matmul(&input, &w_q, seq_len, hidden_size, hidden_size);
    let k = matmul(&input, &w_k, seq_len, hidden_size, hidden_size);
    let v = matmul(&input, &w_v, seq_len, hidden_size, hidden_size);

    let k_transpose = transpose(&k, seq_len, hidden_size);

    let scores = matmul(&q, &k_transpose, seq_len, hidden_size, seq_len);
    let attention = softmax(&scores, seq_len, seq_len);

    // apply attention
    let output = matmul(&attention, &v, seq_len, seq_len, hidden_size);

    output
}

pub fn split_heads(
    input: &[f32],
    seq_len: usize,
    hidden_size: usize,
    num_heads: usize,
) -> Vec<Vec<f32>> {
    let mut heads: Vec<Vec<f32>> = Vec::new();
    let head_size = hidden_size / num_heads;

    for head_idx in 0..num_heads {
        let mut head_data: Vec<f32> = Vec::new();

        let start_feature = head_idx * head_size;
        let end_feature = start_feature + head_size;

        for word_idx in 0..seq_len {
            let word_start = word_idx * hidden_size;

            for feature_idx in start_feature..end_feature {
                head_data.push(input[word_start + feature_idx]);
            }
        }
        heads.push(head_data);
    }

    heads
}

pub fn multi_head_attention(
    input: &[f32],
    heads: usize,
    seq_len: usize,
    hidden_size: usize,
    head_weights: &[HeadWeights],
) -> Vec<Vec<f32>> {
    assert_eq!(head_weights.len(), heads);

    let head_size = hidden_size / heads;
    let mut attention_outputs: Vec<Vec<f32>> = Vec::new();
    let splitted_heads = split_heads(input, seq_len, hidden_size, heads);

    for (i, head) in head_weights.iter().enumerate() {
        let head_output = simple_attention(
            &splitted_heads[i],
            &head.w_q,
            &head.w_k,
            &head.w_v,
            seq_len,
            head_size,
        );
        attention_outputs.push(head_output);
    }
    attention_outputs
}

pub fn concatenate_heads(head_outputs: &[Vec<f32>], seq_len: usize, head_size: usize) -> Vec<f32> {
    let num_heads = head_outputs.len();
    let total_size = seq_len * head_size * num_heads;
    let mut result = Vec::with_capacity(total_size);

    for word_idx in 0..seq_len {
        for head_idx in 0..num_heads {
            let start = word_idx * head_size;
            let end = start + head_size;

            result.extend_from_slice(&head_outputs[head_idx][start..end]);
        }
    }
    result
}

pub fn complete_multi_head_attention(
    input: &[f32],
    heads: usize,
    seq_len: usize,
    hidden_size: usize,
    head_weights: &[HeadWeights],
    w_output: &[f32],
) -> Vec<f32> {
    let head_outputs = multi_head_attention(input, heads, seq_len, hidden_size, head_weights);

    let head_size = hidden_size / heads;
    let concatenated = concatenate_heads(&head_outputs, seq_len, head_size);

    let final_output = matmul(&concatenated, w_output, seq_len, hidden_size, hidden_size);

    final_output
}
