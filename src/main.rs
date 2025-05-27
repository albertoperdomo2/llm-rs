fn test_complete_phi3_generation() {
    println!("testing with REAL Phi-3 tokenizer!");
    
    let model_dir = "models/phi-3-mini";
    
    match llm_rs::generation::Generator::from_pretrained(model_dir) {
        Ok(generator) => {
            println!("loaded generator with real Phi-3 tokenizer!");
            println!("vocabulary size: {}", generator.model.tokenizer.vocab_size());
            
            let result = generator.generate("The future of AI is", 10);
            match result {
                Ok(text) => println!("generated: '{}'", text),
                Err(e) => println!("generation failed: {}", e),
            }
        },
        Err(e) => println!("failed to load: {}", e),
    }
}

fn main() {
    test_complete_phi3_generation();
}
