//! 缓存管理门面。
//!
//! 调度层只与本模块交互，不直接碰 [`crate::pool`] 和 [`crate::radix`] 的内部结构。
//!
//! 规划接口：
//!
//! - `allocate(seq_id, tokens) -> BlockTable`：前缀树命中，再补分配新块。
//! - `free(seq_id)`：归还引用，触发可回收路径。
//! - `evict(n_blocks)`：从 LRU 队列淘汰，供显存吃紧时调用。
//! - `stats() -> CacheStats`：命中率与块使用率，用于调优和压测。
//!
//! TODO(phase-2): 实现分配与释放。
//! TODO(phase-4): 接入前缀树。
