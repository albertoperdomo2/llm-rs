use std::path::Path;

pub fn download_model(repo_id: &str, local_dir: &str) -> crate::Result<String> {
    // for now, assume model is already downloaded

    let model_path = format!("{}/model.safetensors", local_dir);

    if !Path::new(&model_path).exists() {
        return Err(format!("Model not found at {}. Please download manually for now.", model_path).into());
    }

    Ok(model_path)
}

pub fn download_from_hub(repo_id: &str, filename: &str, local_dir: &str) -> crate::Result<()> {
    todo!("implement HF API calls")
}
