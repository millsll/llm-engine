//! 模型超参数。

use std::path::Path;

use serde::Deserialize;

use crate::error::{Error, Result};

/// 模型超参数。
///
/// 字段名刻意与 HuggingFace `config.json` 保持一致，这样可以不做转换
/// 直接反序列化官方权重目录里的配置文件。
#[derive(Debug, Clone, Deserialize)]
pub struct ModelConfig {
    /// 词表大小。
    pub vocab_size: usize,
    /// 隐藏层维度。
    pub hidden_size: usize,
    /// Transformer 层数。
    pub num_hidden_layers: usize,
    /// Query 头数。
    pub num_attention_heads: usize,
    /// KV 头数。等于 `num_attention_heads` 时为 MHA，小于时为 GQA。
    #[serde(default)]
    pub num_key_value_heads: Option<usize>,
    /// 每头维度。缺省时由 `hidden_size / num_attention_heads` 推导。
    #[serde(default)]
    pub head_dim: Option<usize>,
    /// FFN 中间维度。
    pub intermediate_size: usize,
    /// RMSNorm 的 epsilon。
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f64,
    /// RoPE 基数。
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f64,
    /// 最大位置数。
    #[serde(default = "default_max_position_embeddings")]
    pub max_position_embeddings: usize,
    /// 是否共享输入 embedding 与输出投影。
    #[serde(default)]
    pub tie_word_embeddings: bool,
    /// FFN 激活函数名，如 `silu`。
    #[serde(default = "default_hidden_act")]
    pub hidden_act: String,
}

fn default_rms_norm_eps() -> f64 {
    1e-6
}

fn default_rope_theta() -> f64 {
    10_000.0
}

fn default_max_position_embeddings() -> usize {
    2048
}

fn default_hidden_act() -> String {
    "silu".to_owned()
}

impl ModelConfig {
    /// 从 `config.json` 文件读取。
    pub fn from_json_file(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let text = std::fs::read_to_string(path)
            .map_err(|e| Error::Config(format!("读取配置文件 {} 失败: {e}", path.display())))?;
        Self::from_json_str(&text)
    }

    /// 从 JSON 字符串解析。
    pub fn from_json_str(text: &str) -> Result<Self> {
        serde_json::from_str(text).map_err(|e| Error::Config(format!("解析模型配置失败: {e}")))
    }

    /// 每头维度。配置未显式给出时按 `hidden_size / num_attention_heads` 推导。
    pub fn head_dim(&self) -> Result<usize> {
        match self.head_dim {
            Some(dim) => Ok(dim),
            None => {
                if self.num_attention_heads == 0 {
                    return Err(Error::Config("num_attention_heads 为 0".to_owned()));
                }
                Ok(self.hidden_size / self.num_attention_heads)
            }
        }
    }

    /// KV 头数。未显式给出时退化为 MHA。
    pub fn num_kv_heads(&self) -> usize {
        self.num_key_value_heads.unwrap_or(self.num_attention_heads)
    }

    /// Query 头数相对 KV 头数的复制倍数，GQA 用。
    pub fn num_kv_groups(&self) -> usize {
        self.num_attention_heads / self.num_kv_heads()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const QWEN2_05B: &str = r#"{
        "vocab_size": 151936,
        "hidden_size": 896,
        "num_hidden_layers": 24,
        "num_attention_heads": 14,
        "num_key_value_heads": 2,
        "intermediate_size": 4864,
        "rms_norm_eps": 1e-06,
        "rope_theta": 1000000.0,
        "max_position_embeddings": 32768,
        "tie_word_embeddings": true,
        "hidden_act": "silu"
    }"#;

    #[test]
    fn parses_qwen2_config() {
        let cfg = ModelConfig::from_json_str(QWEN2_05B).unwrap();
        assert_eq!(cfg.num_attention_heads, 14);
        assert_eq!(cfg.num_kv_heads(), 2);
        assert_eq!(cfg.num_kv_groups(), 7);
        assert_eq!(cfg.head_dim().unwrap(), 64);
    }

    #[test]
    fn fills_defaults_for_missing_fields() {
        let cfg = ModelConfig::from_json_str(
            r#"{"vocab_size": 8, "hidden_size": 16, "num_hidden_layers": 1,
                "num_attention_heads": 2, "intermediate_size": 32}"#,
        )
        .unwrap();
        assert_eq!(cfg.num_kv_heads(), 2);
        assert_eq!(cfg.head_dim().unwrap(), 8);
        assert!((cfg.rope_theta - 10_000.0).abs() < f64::EPSILON);
        assert_eq!(cfg.hidden_act, "silu");
    }
}
