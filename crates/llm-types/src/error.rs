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

    /// JSON 解析或序列化失败。
    ///
    /// 直接包住 `serde_json::Error` 而不是拍平成 `String`，是为了保留它带的行列号
    /// ——「第 3 行第 17 列期望一个字符串」这种信息在排查畸形 `config.json` 或
    /// safetensors index 时最有用。调用方若知道是哪个文件，应当在外层补上下文
    /// （serde 的错误本身不含文件名）。
    #[error("JSON 解析失败: {0}")]
    Json(#[from] serde_json::Error),

    /// 张量运算失败。
    #[error(transparent)]
    Tensor(#[from] candle_core::Error),

    /// 文件系统操作失败。
    #[error(transparent)]
    Io(#[from] std::io::Error),
}

/// 便捷 `Result` 别名，默认错误类型为 [`Error`]。
pub type Result<T, E = Error> = std::result::Result<T, E>;

#[cfg(test)]
mod tests {
    use super::*;

    /// `?` 能把 `serde_json::Error` 直接转成 `Error`。
    #[test]
    fn converts_from_serde_json() {
        fn parse(text: &str) -> Result<serde_json::Value> {
            Ok(serde_json::from_str(text)?)
        }

        assert!(parse(r#"{"a": 1}"#).is_ok());
        assert!(parse("{ 这不是 JSON").is_err());
    }

    #[test]
    fn json_error_keeps_location_and_source() {
        let err = parse_failure();

        // 前缀说明是 JSON 的问题，而不是别的
        assert!(err.to_string().contains("JSON"), "{err}");
        // 行列号被保留下来
        assert!(err.to_string().contains("line"), "{err}");
        // 底层错误没有丢，可以继续往上追溯
        assert!(std::error::Error::source(&err).is_some());
    }

    fn parse_failure() -> Error {
        serde_json::from_str::<serde_json::Value>("{\n  \"a\": }")
            .unwrap_err()
            .into()
    }

    /// 原有的 io 转换没有被新变体破坏。
    #[test]
    fn converts_from_io() {
        let err: Error = std::fs::read_to_string("/nonexistent/llm-engine/no-such-file")
            .unwrap_err()
            .into();
        assert!(matches!(err, Error::Io(_)));
    }
}
