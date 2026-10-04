//! 前缀树，对标 SGLang 的 RadixAttention。
//!
//! 规划内容：
//!
//! - 节点保存一段 token 区间和对应的物理块引用，边由 token 序列压缩。
//! - `match_prefix(tokens)` 返回命中长度与可复用的 [`crate::BlockTable`]。
//! - 节点引用计数，计数归零后进入 LRU 淘汰队列。
//! - 淘汰叶子优先，保证被共享的内部节点最后才释放。
//!
//! TODO(phase-4): 实现 `RadixTree`，并补上 cache-aware 调度的最长前缀优先策略。
