//! 可复用的模型层。
//!
//! 规划内容：
//!
//! - `RmsNorm` —— 权重乘在归一化之后，eps 在内部。
//! - `RotaryEmbedding` —— 预计算 cos/sin 表，支持 offset 以适配增量解码。
//! - `Attention` —— GQA：KV 头数少于 query 头数时按组复制；通过
//!   [`llm_types::KvCache`] 读写 KV。
//! - `Mlp` —— SwiGLU：`down(silu(gate(x)) * up(x))`。
//! - `DecoderLayer` —— pre-norm 残差结构。
//!
//! TODO(phase-1): 实现上述层。
