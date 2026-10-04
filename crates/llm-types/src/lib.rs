//! # llm-types
//!
//! 共享的数据类型与 trait 定义，**不含任何计算逻辑**。
//!
//! 存在的意义是让 `llm-core`（模型与前向）和 `mem-cache`（KV 存储）都能看到
//! 同一份契约，同时彼此互不依赖：
//!
//! ```text
//!            llm-types
//!           ↑    ↑    ↑
//!  llm-core ┘    │    └ mem-cache
//! ```
//!
//! ## 放什么、不放什么
//!
//! 放：纯数据（配置、参数、错误）、纯数据结构的 trait（[`KvCache`]）。
//! 不放：算子实现、模型定义、显存分配、调度逻辑。
//!
//! 判断标准：如果一段代码需要 `candle-nn`，或者需要分配显存，它就不该在这里。
//! 依赖 candle-core 是必要的（[`KvCache`] 要用 [`Tensor`]，[`Error`] 要包装
//! candle 的错误），但也仅止于此。
//!
//! [`Tensor`]: candle_core::Tensor

pub mod block;
pub mod config;
pub mod device;
pub mod error;
pub mod kv;
pub mod sampling;

pub use block::{BlockId, BlockTable};
pub use config::ModelConfig;
pub use device::{DTypeSpec, DeviceSpec};
pub use error::{Error, Result};
pub use kv::KvCache;
pub use sampling::SamplingParams;
