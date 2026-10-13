use crate::ModelConfig;
use crate::Result;

pub trait CoreConfig {
    fn get_head_dim(&self) -> usize;
    fn get_max_position_embeddings(&self) -> usize;
    fn get_rope_theta(&self) -> f64;
}

pub struct LlamaConfig {
    /// 词表大小。
    pub vocab_size: usize,
    /// 隐藏层维度。
    pub hidden_size: usize,
    /// Transformer 层数。
    pub num_hidden_layers: usize,
    /// RMSNorm 的 epsilon。
    pub rms_norm_eps: f64,
    /// FFN 中间维度。
    pub intermediate_size: usize,
    /// Query 头数。
    pub num_attention_heads: usize,
    /// KV 头数。等于 `num_attention_heads` 时为 MHA，小于时为 GQA。
    pub num_key_value_heads: usize,
    /// 每头维度，已推导过。
    pub head_dim: usize,
    /// RoPE 基数。
    pub rope_theta: f64,
    /// 最大位置数。
    pub max_position_embeddings: usize,
}

impl CoreConfig for LlamaConfig {
    fn get_head_dim(&self) -> usize {
        self.head_dim
    }
    fn get_max_position_embeddings(&self) -> usize {
        self.max_position_embeddings
    }
    fn get_rope_theta(&self) -> f64 {
        self.rope_theta
    }
}

impl LlamaConfig {
    pub fn from_model_config(config: &ModelConfig) -> Result<Self> {
        Ok(Self {
            vocab_size: config.vocab_size,
            hidden_size: config.hidden_size,
            num_hidden_layers: config.num_hidden_layers,
            rms_norm_eps: config.rms_norm_eps,
            intermediate_size: config.intermediate_size,
            num_attention_heads: config.num_attention_heads,
            num_key_value_heads: config.num_key_value_heads,
            head_dim: config.head_dim,
            rope_theta: config.rope_theta,
            max_position_embeddings: config.max_position_embeddings,
        })
    }
}
