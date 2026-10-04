//! Llama 3.x 模型。
//!
//! 与 Qwen2 的结构差异：QKV 是否带 bias、RoPE scaling 方式、是否 tie embedding。
//!
//! TODO(phase-1): 在 qwen2 数值对齐通过后实现。
