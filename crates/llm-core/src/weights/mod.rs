//! 权重加载。
//!
//! - [`mapping`] —— HuggingFace 权重名与内部参数名的映射。
//! - [`loader`] —— safetensors 分片加载、dtype 转换、VarBuilder 组装。
//!
//! TODO(phase-1): 先支持从本地目录加载；随后接 `hf-hub` 按模型名直接下载。

pub mod loader;
pub mod mapping;
