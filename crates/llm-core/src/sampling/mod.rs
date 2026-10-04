//! 采样实现。
//!
//! 参数类型是 [`llm_types::SamplingParams`]，本模块只负责按参数把 logits 变成
//! 下一个 token。
//!
//! - Phase 1：贪心（`argmax`），用于和 HuggingFace 参考实现逐 token 对齐。
//! - Phase 6：top-k / top-p / min-p、重复与频率惩罚、logit bias、种子控制。

// TODO(phase-1): 实现贪心采样。
