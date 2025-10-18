use std::path::PathBuf;

pub fn get_hub_model_file(
    repo_id: &str,
    sub_folder: Option<&str>,
    file: &str,
) -> anyhow::Result<PathBuf> {
    let file_path = sub_folder
        .map(|f| format!("{}/{}", f, file))
        .unwrap_or(file.to_string());
    let api = hf_hub::api::sync::Api::new()?;
    let repo = api.model(repo_id.to_string());
    let file = repo.get(&file_path)?;
    Ok(file)
}
