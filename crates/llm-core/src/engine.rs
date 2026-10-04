//! 生成循环。
//!
//! Phase 1 是最简路径：单请求、无调度。
//!
//! - prefill：整段 prompt 一次前向，产出末位 logits 并写入 KV cache。
//! - decode：每步只喂 1 个 token，从 KV cache 读历史 KV。
//! - 每步用 [`llm_types::SamplingParams`] 采样出下一个 token。
//!
//! Phase 3 起，这个循环的驱动权交给 `llm-runtime`，本模块退化为
//! 「给定一批序列执行一次前向」的批量执行器。
//!
//! TODO(phase-1): 实现单请求生成循环。
