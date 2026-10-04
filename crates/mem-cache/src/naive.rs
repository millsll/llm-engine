//! 朴素 KV cache：每层一个连续张量。
//!
//! 目的是先把前向传播的正确性跑通，不做任何块管理，追加时直接拼接张量。
//! Phase 2 起由 [`crate::paged::PagedKvCache`] 承担生产路径，两者实现同一个
//! [`llm_types::KvCache`] trait，`llm-core` 的模型代码无需改动。
//!
//! TODO(phase-1): 实现 `NaiveKvCache`。
