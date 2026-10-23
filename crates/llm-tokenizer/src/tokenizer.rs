use llm_types::Result;

pub trait Tokenizer: Send + Sync {
    fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>>;

    fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String>;
}
