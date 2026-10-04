//! 运行设备与数值精度选择。

use candle_core::{DType, Device};

use crate::error::{Error, Result};

/// 运行设备描述，可从命令行参数或环境变量解析。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DeviceSpec {
    /// CPU。
    Cpu,
    /// 指定序号的 CUDA 设备。
    Cuda(usize),
    /// Apple Metal 设备。
    Metal,
}

impl DeviceSpec {
    /// 解析 `cpu` / `cuda` / `cuda:1` / `metal`。
    pub fn parse(spec: &str) -> Result<Self> {
        let spec = spec.trim();
        match spec {
            "cpu" => Ok(Self::Cpu),
            "metal" => Ok(Self::Metal),
            "cuda" => Ok(Self::Cuda(0)),
            other => match other.strip_prefix("cuda:") {
                Some(idx) => idx
                    .parse()
                    .map(Self::Cuda)
                    .map_err(|_| Error::Config(format!("无法解析 CUDA 设备序号: {other}"))),
                None => Err(Error::Config(format!(
                    "未知设备 {other}，可选值: cpu / cuda[:N] / metal"
                ))),
            },
        }
    }

    /// 构造 candle 设备句柄。
    ///
    /// CUDA / Metal 需要对应 feature 已启用且运行环境可用，否则返回错误。
    pub fn build(&self) -> Result<Device> {
        match self {
            Self::Cpu => Ok(Device::Cpu),
            Self::Cuda(index) => Ok(Device::new_cuda(*index)?),
            Self::Metal => Ok(Device::new_metal(0)?),
        }
    }
}

/// 数值精度描述。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DTypeSpec {
    /// 32 位浮点。调试与数值比对用。
    F32,
    /// 16 位浮点。
    F16,
    /// bfloat16。推理默认。
    Bf16,
}

impl DTypeSpec {
    /// 解析 `f32` / `f16` / `bf16`。
    pub fn parse(spec: &str) -> Result<Self> {
        match spec.trim() {
            "f32" | "fp32" => Ok(Self::F32),
            "f16" | "fp16" => Ok(Self::F16),
            "bf16" | "bfloat16" => Ok(Self::Bf16),
            other => Err(Error::Config(format!(
                "未知精度 {other}，可选值: f32 / f16 / bf16"
            ))),
        }
    }

    /// 转为 candle 的 [`DType`]。
    pub fn to_dtype(self) -> DType {
        match self {
            Self::F32 => DType::F32,
            Self::F16 => DType::F16,
            Self::Bf16 => DType::BF16,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_device_specs() {
        assert_eq!(DeviceSpec::parse("cpu").unwrap(), DeviceSpec::Cpu);
        assert_eq!(DeviceSpec::parse("cuda").unwrap(), DeviceSpec::Cuda(0));
        assert_eq!(DeviceSpec::parse("cuda:3").unwrap(), DeviceSpec::Cuda(3));
        assert!(DeviceSpec::parse("tpu").is_err());
        assert!(DeviceSpec::parse("cuda:x").is_err());
    }

    #[test]
    fn parses_dtype_specs() {
        assert_eq!(DTypeSpec::parse("bf16").unwrap().to_dtype(), DType::BF16);
        assert_eq!(DTypeSpec::parse("f32").unwrap().to_dtype(), DType::F32);
        assert!(DTypeSpec::parse("int8").is_err());
    }
}
