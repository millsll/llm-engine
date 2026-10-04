//! 采样参数（纯数据）。
//!
//! 采样算法本身在 `llm-core::sampling`。

/// 采样参数。
#[derive(Debug, Clone, PartialEq)]
pub struct SamplingParams {
    /// 采样温度。`<= 0` 表示贪心解码。
    pub temperature: f32,
    /// nucleus 采样累积概率阈值，`1.0` 表示不启用。
    pub top_p: f32,
    /// top-k 截断，`0` 表示不启用。
    pub top_k: usize,
    /// 单次请求最多生成的 token 数。
    pub max_tokens: usize,
    /// 随机种子。`None` 表示每次运行都不同。
    pub seed: Option<u64>,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 0.0,
            top_p: 1.0,
            top_k: 0,
            max_tokens: 256,
            seed: None,
        }
    }
}

impl SamplingParams {
    /// 是否走贪心路径。
    pub fn is_greedy(&self) -> bool {
        self.temperature <= 0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_greedy() {
        let params = SamplingParams::default();
        assert!(params.is_greedy());
        assert_eq!(params.max_tokens, 256);
    }
}
