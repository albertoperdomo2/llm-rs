fn test_simple_attention() {
    let seq_len = 3;
    let hidden_size = 2;
    
    // 3 words represented as [3, 2] matrix
    let input = vec![
        1.0, 0.0,  // word 1 
        0.0, 1.0,  // word 2
        1.0, 1.0   // word 3
    ];
    
    let w_q = vec![
        1.0, 0.0,
        0.0, 1.0
    ];
    
    let w_k = vec![
        1.0, 0.0,
        0.0, 1.0
    ];
    
    let w_v = vec![
        1.0, 0.0,
        0.0, 1.0
    ];
    
    let output = llm_rs::simple_attention(&input, &w_q, &w_k, &w_v, seq_len, hidden_size);
    
    println!("Input: {:?}", input);
    println!("Output: {:?}", output);
    
    // basic sanity checks
    assert_eq!(output.len(), seq_len * hidden_size);
    
    // output should be different from input (attention should mix information)
    assert_ne!(output, input);
    
    println!("attention test passed!");
}

fn main() {
    test_simple_attention();
}
