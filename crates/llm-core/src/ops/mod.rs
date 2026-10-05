//! 张量层封装。
//!
//! 全引擎统一从本模块引入 candle 的 [`Tensor`] / [`Device`] / [`DType`]，
//! 将来若要替换或包裹张量层，改动集中在这一处。
//!
//! [`ops`] 存放本项目自己实现的算子；Phase 1 允许先直接用 `candle_nn`
//! 的等价实现把正确性跑通，再逐个替换并补数值对齐测试。

pub mod activation;
pub mod baseops;

pub use baseops::{repeat_kv, rms_norm, silu_and_mul, softmax_last_dim};
pub use candle_core::{D, DType, Device, IndexOp, Tensor};
