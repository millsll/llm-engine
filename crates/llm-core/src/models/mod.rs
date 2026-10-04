//! 模型定义。
//!
//! Phase 1 目标是 Llama 系架构（Qwen2 / Qwen2.5 / Llama 3.x 同源）：
//!
//! - [`layers`] —— 可复用层：RMSNorm、RoPE、GQA Attention、SwiGLU MLP。
//! - [`qwen2`] —— Qwen2 / Qwen2.5。
//! - [`llama`] —— Llama 3.x，与 Qwen2 的差异在 QKV bias 与 RoPE scaling。
//!
//! TODO(phase-1): 实现 layers 与 qwen2，跑通单请求贪心生成。

pub mod llama;
pub mod qwen2;
