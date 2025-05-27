use crate::math::{matmul, silu};

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
