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
