//! 分页 KV cache。
//!
//! 实现 [`llm_types::KvCache`]：每条序列持有一个 [`crate::BlockTable`]，
//! token `i` 落在第 `i / block_size` 个逻辑块的块内偏移 `i % block_size` 处。
//!
//! `KvCache::get` 需要按块表做一次 gather 才能返回连续张量。这个开销在
//! Phase 3 之后由自定义 paged-attention 算子消除，届时通过 trait 的可选块表
//! 访问器直接按块读取。
//!
//! TODO(phase-2): 实现 `PagedKvCache`，并复用 `llm-core` 的模型测试验证等价性。
