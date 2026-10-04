//! 引擎统一错误类型。

use candle_core;
use thiserror::Error;

/// 引擎统一错误。
#[derive(Debug, Error)]
pub enum Error {
    /// 权重文件缺失、损坏或命名对不上。
    #[error("权重加载失败: {0}")]
    WeightLoad(String),

    /// 模型配置缺失或取值非法。
    #[error("模型配置非法: {0}")]
    Config(String),

    /// 调用方传入的请求参数非法。
    #[error("请求参数非法: {0}")]
    InvalidRequest(String),

    /// KV cache 分配、复用或释放出错。
    #[error("KV cache 错误: {0}")]
    Cache(String),

    /// 张量运算失败。
    #[error(transparent)]
    Tensor(#[from] candle_core::Error),

    /// 文件系统操作失败。
    #[error(transparent)]
    Io(#[from] std::io::Error),
}

/// 便捷 `Result` 别名，默认错误类型为 [`Error`]。
pub type Result<T, E = Error> = std::result::Result<T, E>;
