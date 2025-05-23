#![feature(portable_simd)]

use std::simd::{Simd, f32x8};
// f32x8 works on most CPUs but ideally f32x16 would be better

pub fn matmul(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
    // a is mxn, b is nxk and c is mxk
    // all matrices are stored row-major as flat vectors
    const SIMD_WIDTH: usize = 8; // process 8 f32s at once
    let mut vec = vec![0.0; m * k];

    // transpose B for more straight access
    let b_transposed = transpose(b, n, k);

    if n < SIMD_WIDTH {
        for i in 0..m {
            for j in 0..k {
                let mut sum = 0.0;
                for l in 0..n {
                    sum += a[i * n + l] * b_transposed[j * n + l];
                }
                vec[i * k + j] = sum;
            }
        }
        return vec;
    }

    for i in 0..m {
        for j in 0..k {
            let mut sum = f32x8::splat(0.0);
            let a_row = &a[i * n..(i + 1) * n];
            let b_row = &b_transposed[j * n..(j + 1) * n];

            // process 8 elements at once
            let chunks = n / SIMD_WIDTH;
            for chunk in 0..chunks {
                let start = chunk * SIMD_WIDTH;
                let a_vec = f32x8::from_slice(&a_row[start..start + SIMD_WIDTH]);
                let b_vec = f32x8::from_slice(&b_row[start..start + SIMD_WIDTH]);
                sum += a_vec * b_vec;
            }

            let sum_array: [f32; 8] = sum.into();
            let mut final_sum = sum_array.iter().sum::<f32>();

            // handle remaining elements
            for l in (chunks * SIMD_WIDTH)..n {
                final_sum += a_row[l] * b_row[l];
            }

            vec[i * k + j] = final_sum;
        }
    }

    vec
}

pub fn transpose(matrix: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut transposed = vec![0.0; rows * cols];
    for i in 0..cols {
        for j in 0..rows {
            transposed[i * rows + j] = matrix[j * cols + i];
        }
    }

    transposed
}

pub fn softmax(scores: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut softmax_matrix = vec![0.0; rows * cols];

    for i in 0..rows {
        let score_vector = &scores[i * cols..i * cols + cols];
        let max_val = score_vector.iter().cloned().reduce(f32::max).unwrap_or(0.0);

        let mut softmax_row: Vec<f32> = score_vector.iter().map(|&x| (x - max_val).exp()).collect();

        let sum: f32 = softmax_row.iter().sum();

        for x in &mut softmax_row {
            *x /= sum;
        }

        softmax_matrix[i * cols..i * cols + cols].copy_from_slice(&softmax_row);
    }

    softmax_matrix
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

pub struct HeadWeights {
    pub w_q: Vec<f32>,
    pub w_k: Vec<f32>,
    pub w_v: Vec<f32>,
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

pub fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())  // x * sigmoid(x)
}

pub fn swiglu_feedforward(
    input: &[f32],
    w_gate: &[f32],
    w_up: &[f32],
    w_down: &[f32],
    seq_len: usize,
    hidden_size: usize,
    intermediate_size: usize,
) -> Vec<f32> {
    // SwiGLU(x) = SiLU(x × W_gate) (dot) (x × W_up) × W_down
    let gate_linear = matmul(input, w_gate, seq_len, hidden_size, intermediate_size);

    let mut gate_activated = Vec::new();
    for &x in gate_linear.iter() {
        gate_activated.push(silu(x));
    }

    let up = matmul(input, w_up, seq_len, hidden_size, intermediate_size);

    let combined: Vec<f32> = gate_activated.iter()
        .zip(up.iter())
        .map(|(x, y)| x * y)
        .collect();

    let output = matmul(&combined, w_down, seq_len, intermediate_size, hidden_size);

    output
}

pub fn rms_norm(input: &[f32], weight: &[f32], seq_len: usize, hidden_size: usize) -> Vec<f32> {
    let epsilon = 1e-5;
    let mut output = Vec::with_capacity(input.len());

    for word_idx in 0..seq_len {
        let word_features = &input[word_idx * hidden_size..word_idx * hidden_size + hidden_size];

        let rms = (word_features.iter().map(|x| x * x).sum::<f32>() / hidden_size as f32).sqrt() + epsilon;

        for (i, &feature) in word_features.iter().enumerate() {
            let normalized = (feature / rms) * weight[i];
            output.push(normalized);
        }
    }

    output
}

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

pub mod tokenizer;
