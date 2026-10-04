//! Qwen2 / Qwen2.5 模型。
//!
//! Phase 1 的首个目标架构。用 Qwen2.5-0.5B 做端到端对齐：体积小，
//! 单机 CPU 就能跑，便于快速迭代。
//!
//! TODO(phase-1): 实现 `Model::forward(tokens, positions, cache) -> logits`。
