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

pub fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())  // x * sigmoid(x)
}