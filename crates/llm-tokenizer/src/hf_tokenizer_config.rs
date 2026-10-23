use std::path::Path;

use llm_types::Result;
use serde;
use serde::Deserialize;
use serde_json;

#[derive(Debug, Default, Deserialize)]
#[serde(default)]
pub struct HfTokenizerConfig {
    pub bos_token: Option<String>,
    pub eos_token: Option<String>,
    pub unk_token: Option<String>,
    pub pad_token: Option<String>,

    pub add_bos_token: bool,
    pub add_eos_token: bool,
}

impl HfTokenizerConfig {
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let file = std::fs::File::open(path)?;
        let config = serde_json::from_reader(file)?;
        Ok(config)
    }
}
