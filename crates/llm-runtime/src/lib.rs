//! # llm-runtime
//!
//! 运行时编排层。它是唯一同时认识 `llm-core` 和 `mem-cache` 的地方，
//! 因此也是唯一负责把两者接起来的地方：
//!
//! ```text
//! 请求 -> 分词 -> 分配 KV 块 -> 组批 -> 调用 llm-core 前向
//!                                              |
//!      <- 反分词 <- 采样 <--------------------- +
//! ```
//!
//! ## 为什么编排放在这里
//!
//! `llm-core` 不认识 cache，`mem-cache` 不认识模型。两边都只认识
//! `llm-types` 里的契约。谁来决定"这条序列用分页 cache、块表长这样、
//! 下一步算哪些 token"？只有运行时层知道答案。
//!
//! 这也让 `llm-core` 的前向签名保持纯粹：只接收
//! `&mut dyn KvCache` 和「数据」（token、位置、块表），不接收「策略」。
//!
//! ## 模块
//!
//! - [`scheduler`]：准入、连续批处理组批、chunked prefill、抢占。
//!
//! TODO(phase-3): 实现 FCFS 组批与重算式抢占。

pub mod scheduler;
