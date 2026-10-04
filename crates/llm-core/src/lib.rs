//! # llm-core
//!
//! 模型结构与前向计算。职责：
//!
//! - **模型定义**：embedding、RMSNorm、RoPE、GQA attention、SwiGLU MLP。
//! - **权重加载**：safetensors 分片加载与 HuggingFace 权重名映射。
//! - **采样实现**：greedy / top-k / top-p（参数类型见 [`llm_types::SamplingParams`]）。
//! - **单请求生成循环**：[`engine`]；批量调度从 Phase 3 起交给 `llm-runtime`。
//!
//! 张量层使用 candle，本 crate 不实现自己的 CUDA kernel。
//!
//! ## 与其他 crate 的关系
//!
//! - 数据类型与 trait 来自 [`llm_types`]，本 crate 不重复定义。
//! - **不依赖 `mem-cache`**：前向只接收 `&mut dyn KvCache`，具体是朴素实现
//!   还是分页实现由调用方决定。
//! - 不依赖 `llm-runtime`：驱动权在运行时层，前向只被调用。

pub mod engine;
pub mod models;
pub mod sampling;
pub mod tensor;
pub mod weights;

pub use llm_types::{Error, ModelConfig, Result};
