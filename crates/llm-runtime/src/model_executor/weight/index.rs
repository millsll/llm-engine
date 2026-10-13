use llm_types::Result;
use serde::Deserialize;
use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::path::PathBuf;

#[derive(Debug, Deserialize)]
struct SafetensorIndex {
    pub metadata: Metadata,
    pub weight_map: HashMap<String, String>,
}
#[derive(Debug, Deserialize)]
struct Metadata {
    pub total_size: u64,
}

impl SafetensorIndex {
    pub fn shard_files(&self) -> Vec<&str> {
        let mut files = self
            .weight_map
            .values()
            .map(String::as_str)
            .collect::<HashSet<_>>()
            .into_iter()
            .collect::<Vec<_>>();
        files.sort_unstable();
        files
    }
}

pub fn find_safetensors_files(mode_dir: &Path) -> Result<Vec<PathBuf>> {
    let index_name = "model.safetensors.index.json";
    let index: SafetensorIndex =
        serde_json::from_reader(std::fs::File::open(mode_dir.join(index_name))?)?;
    let shard_files: Vec<PathBuf> = index
        .shard_files()
        .iter()
        .map(|file_name| mode_dir.join(file_name))
        .collect();
    Ok(shard_files)
}
