//! # mem-cache
//!
//! KV cache 的全部实现。四件事：
//!
//! 1. 朴素 cache（[`naive`]）：每层一段连续张量。Phase 1 用来跑通前向。
//! 2. 显存块池（[`pool`]）：启动时按 `num_blocks` 预分配，运行期只做
//!    分配与回收，不做 `cudaMalloc`。块大小固定为 `block_size` 个 token。
//! 3. 分页 cache（[`paged`]）：实现 [`llm_types::KvCache`]，按块表寻址。
//! 4. 前缀树（[`radix`]）：把 token 序列的前缀映射到物理块，让共享前缀的
//!    请求直接复用已算好的 KV，对标 SGLang 的 RadixAttention。
//!
//! ## 依赖方向
//!
//! 本 crate 只依赖 `llm-types`（契约）和 `candle-core`（张量），
//! **不依赖 `llm-core`**。存储层不需要知道模型长什么样。
//! 因此 `cargo test -p mem-cache` 不会拖上权重加载、HTTP 那一整套依赖。
//!
//! ## 实现顺序
//!
//! Phase 1 做 [`naive`]；Phase 2 做 [`pool`] 与 [`paged`]；Phase 4 做 [`radix`]。

pub mod manager;
pub mod naive;
pub mod paged;
pub mod pool;
pub mod radix;

pub use llm_types::{BlockId, BlockTable, KvCache};
