//! safetensors 权重加载。
//!
//! 规划内容：
//!
//! - 读取 `model.safetensors.index.json`，支持多分片。
//! - 用 `memmap2` 惰性映射，避免一次性把权重读进内存。
//! - dtype 转换：权重可能是 f16 / bf16 / f32，需要统一到运行精度。
//! - 组装成 candle 的 VarBuilder 供模型构造使用。
//!
//! TODO(phase-1): 实现单分片加载，多分片随后补齐。
