use llm_types::Error;
use llm_types::Result;
use std::path::Path;

use tokenizers::Tokenizer;

use crate::tokenizer;

pub struct HfTokenizer {
    inner: Tokenizer,
}

impl HfTokenizer {
    pub fn from_file(tokenizer_path: impl AsRef<Path>) -> Result<Self> {
        let inner =
            tokenizers::Tokenizer::from_file(tokenizer_path.as_ref()).map_err(|source| {
                Error::TokenizerLoad {
                    path: tokenizer_path.as_ref().to_path_buf(),
                    source,
                }
            })?;

        Ok(Self { inner })
    }
}

impl tokenizer::Tokenizer for HfTokenizer {
    fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>> {
        let encoding = self.inner.encode(text, add_special_tokens)?;
        Ok(encoding.get_ids().to_vec())
    }

    fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String> {
        let decoding_res = self.inner.decode(ids, skip_special_tokens)?;
        Ok(decoding_res)
    }
}
