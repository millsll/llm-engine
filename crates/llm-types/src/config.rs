//! 模型配置。
//!
//! 两次分层，各管一件事。
//!
//! ## 第一层：文件格式
//!
//! - [`PretrainedConfig`] —— HuggingFace `config.json` 的 serde 映射。
//!   字段名与文件里的键一一对应，保持宽松：该是 `Option` 的就是 `Option`，
//!   缺省值用 `#[serde(default)]` 填。它只负责「把文件读进来」，不做校验。
//! - [`RuntimeParams`] —— 与模型文件无关、由部署方决定的那部分（设备、精度）。
//!
//! ## 第二层：引擎视图
//!
//! [`ModelConfig`] 把上面两者**和模型目录路径**捏在一起，作为「一个待加载的
//! 模型」的完整描述：
//!
//! ```text
//!   ModelConfig
//!     ├── model_path   模型目录，权重与 config.json 都在这里
//!     ├── pretrained   从 {model_path}/config.json 读出来的原始配置
//!     ├── runtime      设备 / 精度
//!     └── ...          解析后的字段：head_dim、num_key_value_heads 等
//! ```
//!
//! 这样 `ModelLoader` 拿到一个 `ModelConfig` 就知道「去哪读权重、按什么精度读」，
//! 不用再单独传路径。同时字段都是**已解析的结果**——`num_key_value_heads`
//! 已按 MHA 补过默认值，`head_dim` 已推导过，下游不必处理 `Option`，也不会除零。
//!
//! 校验在构造期一次性做完，非法组合在加载模型时就报出来，而不是等前向跑到
//! 注意力层才发现 `head_dim` 推导不出整数。

use std::path::{Path, PathBuf};

use serde::Deserialize;

use crate::device::{DTypeSpec, DeviceSpec};
use crate::error::{Error, Result};

/// 模型目录下配置文件的固定文件名。
const CONFIG_FILE: &str = "config.json";

fn default_rms_norm_eps() -> f64 {
    1e-6
}

fn default_rope_theta() -> f64 {
    10_000.0
}

fn default_max_position_embeddings() -> usize {
    2048
}

fn default_hidden_act() -> String {
    "silu".to_owned()
}

// ---------------------------------------------------------------------------
// PretrainedConfig
// ---------------------------------------------------------------------------

/// HuggingFace `config.json` 的 serde 映射。
///
/// 字段名刻意与配置文件保持一致，这样可以不做转换直接反序列化官方权重目录里的
/// `config.json`。未出现在本结构体里的键会被 serde 静默忽略——HF 的配置里
/// 有大量模型特有的字段（`attention_bias`、`sliding_window` 等），
/// 用 `deny_unknown_fields` 会当场炸掉。
///
/// 这一层**不做校验**：`num_attention_heads` 可以是 0，`head_dim` 可以推不出
/// 整数，`rope_theta` 可以是负数。要拿保证请用 [`ModelConfig`]。
///
/// ```
/// use llm_types::PretrainedConfig;
///
/// let cfg = PretrainedConfig::from_json_str(
///     r#"{"vocab_size": 8, "hidden_size": 16, "num_hidden_layers": 1,
///         "num_attention_heads": 2, "intermediate_size": 32}"#,
/// )
/// .unwrap();
///
/// assert_eq!(cfg.num_kv_heads(), 2);       // 缺省退化成 MHA
/// assert_eq!(cfg.head_dim().unwrap(), 8);  // 由 hidden_size 推导
/// ```
#[derive(Debug, Clone, Deserialize)]
pub struct PretrainedConfig {
    /// 词表大小。
    pub vocab_size: usize,
    /// 隐藏层维度。
    pub hidden_size: usize,
    /// Transformer 层数。
    pub num_hidden_layers: usize,
    /// Query 头数。
    pub num_attention_heads: usize,
    /// KV 头数。等于 `num_attention_heads` 时为 MHA，小于时为 GQA。
    #[serde(default)]
    pub num_key_value_heads: Option<usize>,
    /// 每头维度。缺省时由 `hidden_size / num_attention_heads` 推导。
    #[serde(default)]
    pub head_dim: Option<usize>,
    /// FFN 中间维度。
    pub intermediate_size: usize,
    /// RMSNorm 的 epsilon。
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f64,
    /// RoPE 基数。
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f64,
    /// 最大位置数。
    #[serde(default = "default_max_position_embeddings")]
    pub max_position_embeddings: usize,
    /// 是否共享输入 embedding 与输出投影。
    #[serde(default)]
    pub tie_word_embeddings: bool,
    /// FFN 激活函数名，如 `silu`。
    #[serde(default = "default_hidden_act")]
    pub hidden_act: String,
    /// 模型架构名列表，如 `["Qwen2ForCausalLM"]`。模型加载分发用。
    ///
    /// 用 `Option<Vec<String>>` 而不是 `Vec<String>`，是因为 HF 允许这个键
    /// 缺失**或者为 `null`**——`PretrainedConfig.to_dict()` 在
    /// `architectures=None` 时就会写出 `null`，而 serde 的 `default`
    /// 对付不了显式的 `null`。
    ///
    /// 分发时通常只关心第一个，见 [`PretrainedConfig::arch_name`]。
    #[serde(default)]
    pub architectures: Option<Vec<String>>,
}

impl PretrainedConfig {
    /// 从 JSON 字符串解析。
    pub fn from_json_str(text: &str) -> Result<Self> {
        serde_json::from_str(text).map_err(|e| Error::Config(format!("解析模型配置失败: {e}")))
    }

    /// 从 `config.json` 文件读取。
    pub fn from_json_file(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let text = std::fs::read_to_string(path)
            .map_err(|e| Error::Config(format!("读取配置文件 {} 失败: {e}", path.display())))?;
        Self::from_json_str(&text)
    }

    /// 每头维度。配置未显式给出时按 `hidden_size / num_attention_heads` 推导。
    pub fn head_dim(&self) -> Result<usize> {
        match self.head_dim {
            Some(dim) => Ok(dim),
            None => {
                if self.num_attention_heads == 0 {
                    return Err(Error::Config("num_attention_heads 为 0".to_owned()));
                }
                Ok(self.hidden_size / self.num_attention_heads)
            }
        }
    }

    /// KV 头数。未显式给出时退化为 MHA。
    pub fn num_kv_heads(&self) -> usize {
        self.num_key_value_heads.unwrap_or(self.num_attention_heads)
    }

    /// Query 头数相对 KV 头数的复制倍数，GQA 用。
    ///
    /// 注意这一层不做校验，`num_attention_heads` 为 0 时会除零 panic。
    /// 需要保证请走 [`ModelConfig`]。
    pub fn num_kv_groups(&self) -> usize {
        self.num_attention_heads / self.num_kv_heads()
    }

    /// 首选架构名，取 [`Self::architectures`] 的第一个元素。
    ///
    /// 键缺失、为 `null` 或空数组时返回 `None`，调用方据此回退到别的分发依据。
    pub fn arch_name(&self) -> Option<&str> {
        self.architectures.as_ref()?.first().map(String::as_str)
    }
}

// ---------------------------------------------------------------------------
// RuntimeParams
// ---------------------------------------------------------------------------

/// 与模型文件无关、由部署方决定的运行参数。
///
/// 这些**不能**从 `config.json` 读出来：同一份权重可以跑在 CPU 或 GPU、
/// 用 f32 或 bf16，取决于部署环境。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuntimeParams {
    /// 运行设备。
    pub device: DeviceSpec,
    /// 运行精度。
    pub dtype: DTypeSpec,
}

impl Default for RuntimeParams {
    fn default() -> Self {
        Self {
            // 默认 CPU + f32：无 GPU 的机器也能跑，且便于与 HF 做数值对齐。
            device: DeviceSpec::Cpu,
            dtype: DTypeSpec::F32,
        }
    }
}

// ---------------------------------------------------------------------------
// ModelConfig
// ---------------------------------------------------------------------------

/// 一个待加载模型的完整描述。
///
/// 由「模型目录路径 + 该目录下的 `config.json` + 运行参数」构成。
/// [`ModelConfig::from_model_path`] 负责读文件，其余字段是解析并校验后的结果。
///
/// 字段都是**已解析的值**而不是原始配置：`num_key_value_heads` 已按 MHA
/// 补过默认值，`head_dim` 已推导过，所以下游不需要处理 `Option`，也不会除零。
///
/// 想读原始配置里的其余字段，经 [`ModelConfig::pretrained`] 取。
#[derive(Debug, Clone)]
pub struct ModelConfig {
    /// 模型目录，权重文件与 `config.json` 都在这里。
    model_path: PathBuf,
    /// `{model_path}/config.json` 的 serde 映射。
    pretrained: PretrainedConfig,
    /// 设备与精度。
    runtime: RuntimeParams,

    /// 词表大小。
    pub vocab_size: usize,
    /// 隐藏层维度。
    pub hidden_size: usize,
    /// Transformer 层数。
    pub num_hidden_layers: usize,
    /// Query 头数。
    pub num_attention_heads: usize,
    /// KV 头数，已按 MHA 补过默认值。
    pub num_key_value_heads: usize,
    /// 每头维度，已推导过。
    pub head_dim: usize,
    /// FFN 中间维度。
    pub intermediate_size: usize,
    /// RMSNorm 的 epsilon。
    pub rms_norm_eps: f64,
    /// RoPE 基数。
    pub rope_theta: f64,
    /// 最大位置数。
    pub max_position_embeddings: usize,
    /// 是否共享输入 embedding 与输出投影。
    pub tie_word_embeddings: bool,
    /// FFN 激活函数名，如 `silu`。
    pub hidden_act: String,
}

impl ModelConfig {
    /// 从模型目录加载，使用 [`RuntimeParams::default`]。
    ///
    /// 读取 `{model_path}/config.json`。
    pub fn from_model_path(model_path: impl Into<PathBuf>) -> Result<Self> {
        Self::from_model_path_with(model_path, RuntimeParams::default())
    }

    /// 从模型目录加载，并指定运行参数。
    pub fn from_model_path_with(
        model_path: impl Into<PathBuf>,
        runtime: RuntimeParams,
    ) -> Result<Self> {
        let model_path = model_path.into();
        let pretrained = PretrainedConfig::from_json_file(model_path.join(CONFIG_FILE))?;
        Self::from_parts(model_path, pretrained, runtime)
    }

    /// 不做任何文件 IO，直接用已解析好的内容构造。测试与程序化构造用。
    ///
    /// `model_path` 只是原样记下来，不会被访问。
    pub fn from_parts(
        model_path: impl Into<PathBuf>,
        pretrained: PretrainedConfig,
        runtime: RuntimeParams,
    ) -> Result<Self> {
        let model_path = model_path.into();
        let num_attention_heads = pretrained.num_attention_heads;

        if num_attention_heads == 0 {
            return Err(Error::Config("num_attention_heads 为 0".to_owned()));
        }

        // head_dim 优先用显式值。有些模型（如 Gemma2）会显式给出与
        // hidden_size / num_attention_heads 不同的 head_dim，此时两者不必整除。
        let head_dim = match pretrained.head_dim {
            Some(dim) => dim,
            None => {
                if pretrained.hidden_size % num_attention_heads != 0 {
                    return Err(Error::Config(format!(
                        "hidden_size({}) 不能被 num_attention_heads({num_attention_heads}) 整除，\
                         且配置里没有显式给出 head_dim",
                        pretrained.hidden_size
                    )));
                }
                pretrained.hidden_size / num_attention_heads
            }
        };

        if head_dim == 0 {
            return Err(Error::Config("head_dim 为 0".to_owned()));
        }
        if head_dim % 2 != 0 {
            // rotate_half 式的 RoPE 把 head_dim 前后两半配对，奇数维无解。
            return Err(Error::Config(format!(
                "head_dim({head_dim}) 必须是偶数，否则无法应用 RoPE"
            )));
        }

        let num_key_value_heads = pretrained.num_kv_heads();
        if num_key_value_heads == 0 {
            return Err(Error::Config("num_key_value_heads 为 0".to_owned()));
        }
        if num_attention_heads % num_key_value_heads != 0 {
            return Err(Error::Config(format!(
                "num_attention_heads({num_attention_heads}) 不能被 \
                 num_key_value_heads({num_key_value_heads}) 整除，无法做 GQA 分组"
            )));
        }

        if pretrained.rope_theta <= 0.0 {
            return Err(Error::Config(format!(
                "rope_theta({}) 必须为正，否则 RoPE 频率无意义",
                pretrained.rope_theta
            )));
        }

        Ok(Self {
            vocab_size: pretrained.vocab_size,
            hidden_size: pretrained.hidden_size,
            num_hidden_layers: pretrained.num_hidden_layers,
            num_attention_heads,
            num_key_value_heads,
            head_dim,
            intermediate_size: pretrained.intermediate_size,
            rms_norm_eps: pretrained.rms_norm_eps,
            rope_theta: pretrained.rope_theta,
            max_position_embeddings: pretrained.max_position_embeddings,
            tie_word_embeddings: pretrained.tie_word_embeddings,
            hidden_act: pretrained.hidden_act.clone(),
            model_path,
            pretrained,
            runtime,
        })
    }

    /// 模型目录。
    pub fn model_path(&self) -> &Path {
        &self.model_path
    }

    /// `config.json` 的完整路径。出错信息与权重加载都会用到。
    pub fn config_json_path(&self) -> PathBuf {
        self.model_path.join(CONFIG_FILE)
    }

    /// 底层原始配置。
    pub fn pretrained(&self) -> &PretrainedConfig {
        &self.pretrained
    }

    /// 运行参数。
    pub fn runtime(&self) -> &RuntimeParams {
        &self.runtime
    }

    /// 首选架构名，透传自 [`PretrainedConfig`]。模型加载分发用。
    ///
    /// 返回 `&str` 而不是 `String`，分发时只做字符串比较，不需要分配。
    ///
    /// ```
    /// use llm_types::{ModelConfig, PretrainedConfig, RuntimeParams};
    ///
    /// let pretrained = PretrainedConfig::from_json_str(
    ///     r#"{"architectures": ["Qwen2ForCausalLM"], "vocab_size": 8, "hidden_size": 16,
    ///         "num_hidden_layers": 1, "num_attention_heads": 2, "intermediate_size": 32}"#,
    /// )
    /// .unwrap();
    /// let cfg = ModelConfig::from_parts("/models/qwen2", pretrained, RuntimeParams::default())
    ///     .unwrap();
    ///
    /// match cfg.arch_name() {
    ///     Some("Qwen2ForCausalLM") => {}
    ///     Some(other) => panic!("不支持的架构: {other}"),
    ///     None => {}
    /// }
    /// ```
    pub fn arch_name(&self) -> Option<&str> {
        self.pretrained.arch_name()
    }

    /// Query 头数相对 KV 头数的复制倍数，GQA 用。
    ///
    /// 构造时已保证 `num_attention_heads` 能被 `num_key_value_heads` 整除，
    /// 所以这里不会除零，也不必返回 `Result`。
    pub fn num_kv_groups(&self) -> usize {
        self.num_attention_heads / self.num_key_value_heads
    }

    /// 是否属于 GQA（KV 头数少于 Query 头数）。
    pub fn is_gqa(&self) -> bool {
        self.num_key_value_heads < self.num_attention_heads
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const QWEN2_05B: &str = r#"{
        "architectures": ["Qwen2ForCausalLM"],
        "vocab_size": 151936,
        "hidden_size": 896,
        "num_hidden_layers": 24,
        "num_attention_heads": 14,
        "num_key_value_heads": 2,
        "intermediate_size": 4864,
        "rms_norm_eps": 1e-06,
        "rope_theta": 1000000.0,
        "max_position_embeddings": 32768,
        "tie_word_embeddings": true,
        "hidden_act": "silu"
    }"#;

    /// 只给必需字段，其余全靠 serde 默认值。
    const MINIMAL: &str = r#"{
        "vocab_size": 8,
        "hidden_size": 16,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "intermediate_size": 32
    }"#;

    /// 拼一个最小合法配置，可在末尾追加额外键（`extra` 以逗号开头或为空）。
    fn minimal_json(extra: &str) -> String {
        format!(
            r#"{{"vocab_size": 8, "hidden_size": 16, "num_hidden_layers": 1,
                 "num_attention_heads": 2, "intermediate_size": 32{extra}}}"#
        )
    }

    /// 从 JSON 文本造一个 ModelConfig，不碰文件系统。
    fn model_config(text: &str) -> Result<ModelConfig> {
        ModelConfig::from_parts(
            "/dummy",
            PretrainedConfig::from_json_str(text)?,
            RuntimeParams::default(),
        )
    }

    // ---- PretrainedConfig ----

    #[test]
    fn pretrained_parses_qwen2() {
        let cfg = PretrainedConfig::from_json_str(QWEN2_05B).unwrap();

        assert_eq!(cfg.num_attention_heads, 14);
        assert_eq!(cfg.num_key_value_heads, Some(2));
        assert_eq!(cfg.num_kv_heads(), 2);
        assert_eq!(cfg.num_kv_groups(), 7);
        // 配置里没有 head_dim，由 hidden_size / num_attention_heads 推导
        assert_eq!(cfg.head_dim, None);
        assert_eq!(cfg.head_dim().unwrap(), 64);
        assert_eq!(cfg.max_position_embeddings, 32768);
        assert!(cfg.tie_word_embeddings);
        assert_eq!(cfg.arch_name(), Some("Qwen2ForCausalLM"));
    }

    #[test]
    fn arch_name_handles_missing_null_empty_and_multi() {
        // 缺失
        let cfg = PretrainedConfig::from_json_str(&minimal_json("")).unwrap();
        assert_eq!(cfg.architectures, None);
        assert_eq!(cfg.arch_name(), None);

        // 显式为 null——HF 在 architectures=None 时就是这么写的
        let cfg =
            PretrainedConfig::from_json_str(&minimal_json(r#", "architectures": null"#)).unwrap();
        assert_eq!(cfg.architectures, None);
        assert_eq!(cfg.arch_name(), None);

        // 空数组
        let cfg =
            PretrainedConfig::from_json_str(&minimal_json(r#", "architectures": []"#)).unwrap();
        assert_eq!(cfg.arch_name(), None);

        // 多个：取第一个
        let cfg = PretrainedConfig::from_json_str(&minimal_json(
            r#", "architectures": ["Qwen2ForCausalLM", "Qwen2Model"]"#,
        ))
        .unwrap();
        assert_eq!(cfg.arch_name(), Some("Qwen2ForCausalLM"));
    }

    #[test]
    fn pretrained_fills_defaults() {
        let cfg = PretrainedConfig::from_json_str(MINIMAL).unwrap();

        assert_eq!(cfg.num_key_value_heads, None);
        assert_eq!(cfg.num_kv_heads(), 2);
        assert_eq!(cfg.num_kv_groups(), 1);
        assert_eq!(cfg.head_dim().unwrap(), 8);
        assert!((cfg.rms_norm_eps - 1e-6).abs() < f64::EPSILON);
        assert!((cfg.rope_theta - 10_000.0).abs() < f64::EPSILON);
        assert_eq!(cfg.max_position_embeddings, 2048);
        assert!(!cfg.tie_word_embeddings);
        assert_eq!(cfg.hidden_act, "silu");
    }

    #[test]
    fn pretrained_ignores_unknown_fields() {
        // HF 的 config 里有大量模型特有的键，不能被拒掉
        let cfg = PretrainedConfig::from_json_str(
            r#"{"vocab_size": 8, "hidden_size": 16, "num_hidden_layers": 1,
                "num_attention_heads": 2, "intermediate_size": 32,
                "attention_bias": true, "sliding_window": null,
                "rope_scaling": {"type": "linear", "factor": 2.0}}"#,
        )
        .unwrap();
        assert_eq!(cfg.hidden_size, 16);
    }

    #[test]
    fn pretrained_reports_missing_required_field() {
        let err = PretrainedConfig::from_json_str(
            r#"{"vocab_size": 8, "hidden_size": 16, "num_hidden_layers": 1,
                "num_attention_heads": 2}"#,
        )
        .unwrap_err();
        assert!(err.to_string().contains("intermediate_size"), "{err}");
    }

    #[test]
    fn pretrained_head_dim_errors_on_zero_heads() {
        let cfg = PretrainedConfig::from_json_str(
            r#"{"vocab_size": 8, "hidden_size": 16, "num_hidden_layers": 1,
                "num_attention_heads": 0, "intermediate_size": 32}"#,
        )
        .unwrap();
        // 这一层只报推导不出，不做整体校验
        assert!(cfg.head_dim().is_err());
    }

    // ---- RuntimeParams ----

    #[test]
    fn runtime_params_default_to_cpu_f32() {
        let params = RuntimeParams::default();
        assert_eq!(params.device, DeviceSpec::Cpu);
        assert_eq!(params.dtype, DTypeSpec::F32);
    }

    // ---- ModelConfig：从磁盘加载 ----

    /// 造一个只有 config.json 的临时模型目录。
    fn temp_model_dir(tag: &str, contents: Option<&str>) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("llm_engine_cfg_{tag}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        if let Some(text) = contents {
            std::fs::write(dir.join(CONFIG_FILE), text).unwrap();
        }
        dir
    }

    #[test]
    fn loads_from_model_path() {
        let dir = temp_model_dir("ok", Some(QWEN2_05B));

        let cfg = ModelConfig::from_model_path(&dir).unwrap();

        assert_eq!(cfg.model_path(), dir.as_path());
        assert_eq!(cfg.config_json_path(), dir.join(CONFIG_FILE));
        // 从文件读出来的原始配置
        assert_eq!(cfg.pretrained().num_attention_heads, 14);
        // 解析后的字段
        assert_eq!(cfg.hidden_size, 896);
        assert_eq!(cfg.head_dim, 64);
        assert_eq!(cfg.num_key_value_heads, 2);
        assert_eq!(cfg.arch_name(), Some("Qwen2ForCausalLM"));
        // 默认运行参数
        assert_eq!(cfg.runtime(), &RuntimeParams::default());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn loads_from_model_path_with_runtime_params() {
        let dir = temp_model_dir("runtime", Some(MINIMAL));
        let runtime = RuntimeParams {
            device: DeviceSpec::Cuda(1),
            dtype: DTypeSpec::Bf16,
        };

        let cfg = ModelConfig::from_model_path_with(&dir, runtime.clone()).unwrap();
        assert_eq!(cfg.runtime(), &runtime);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn reports_missing_config_json() {
        // 目录存在但没有 config.json
        let dir = temp_model_dir("absent", None);
        let err = ModelConfig::from_model_path(&dir).unwrap_err();
        // 错误信息里要能看出它找的是哪个文件
        assert!(err.to_string().contains("config.json"), "{err}");
        assert!(
            err.to_string().contains(&dir.display().to_string()),
            "{err}"
        );

        // 目录本身也不存在
        let err = ModelConfig::from_model_path(dir.join("nope")).unwrap_err();
        assert!(err.to_string().contains("config.json"), "{err}");

        let _ = std::fs::remove_dir_all(&dir);
    }

    // ---- ModelConfig：解析与校验 ----

    #[test]
    fn parses_qwen2_config() {
        let cfg = model_config(QWEN2_05B).unwrap();

        assert_eq!(cfg.num_attention_heads, 14);
        assert_eq!(cfg.num_key_value_heads, 2);
        assert_eq!(cfg.num_kv_groups(), 7);
        assert!(cfg.is_gqa());
        assert_eq!(cfg.head_dim, 64);
        assert_eq!(cfg.max_position_embeddings, 32768);
        assert!(cfg.tie_word_embeddings);
        assert_eq!(cfg.arch_name(), Some("Qwen2ForCausalLM"));
        // ModelConfig 的 arch_name 与 PretrainedConfig 的一致
        assert_eq!(cfg.arch_name(), cfg.pretrained().arch_name());
    }

    #[test]
    fn model_config_resolves_options() {
        let cfg = model_config(MINIMAL).unwrap();

        // 没写 num_key_value_heads -> 退化成 MHA
        assert_eq!(cfg.num_key_value_heads, 2);
        assert_eq!(cfg.num_kv_groups(), 1);
        assert!(!cfg.is_gqa());
        // 没写 head_dim -> 由 hidden_size / num_attention_heads 推导
        assert_eq!(cfg.head_dim, 8);
        assert_eq!(cfg.hidden_act, "silu");
    }

    #[test]
    fn explicit_head_dim_wins_over_derived() {
        // Gemma2 形状：hidden_size 不能被 num_attention_heads 整除，
        // 但有显式 head_dim，应当照用不报错。
        let cfg = model_config(
            r#"{"vocab_size": 8, "hidden_size": 24, "num_hidden_layers": 1,
                "num_attention_heads": 8, "head_dim": 4, "intermediate_size": 32}"#,
        )
        .unwrap();
        assert_eq!(cfg.head_dim, 4);
    }

    #[test]
    fn rejects_indivisible_head_dim_derivation() {
        let err = model_config(
            r#"{"vocab_size": 8, "hidden_size": 15, "num_hidden_layers": 1,
                "num_attention_heads": 4, "intermediate_size": 32}"#,
        )
        .unwrap_err();
        assert!(err.to_string().contains("head_dim"), "{err}");
    }

    #[test]
    fn rejects_odd_head_dim() {
        let err = model_config(
            r#"{"vocab_size": 8, "hidden_size": 16, "num_hidden_layers": 1,
                "num_attention_heads": 2, "head_dim": 5, "intermediate_size": 32}"#,
        )
        .unwrap_err();
        assert!(err.to_string().contains("偶数"), "{err}");
    }

    #[test]
    fn rejects_zero_attention_heads() {
        let err = model_config(
            r#"{"vocab_size": 8, "hidden_size": 16, "num_hidden_layers": 1,
                "num_attention_heads": 0, "intermediate_size": 32}"#,
        )
        .unwrap_err();
        assert!(err.to_string().contains("num_attention_heads"), "{err}");
    }

    #[test]
    fn rejects_indivisible_kv_heads() {
        // head_dim 能正常推导（24 / 6 = 4），但 6 不能被 4 整除，GQA 分组无解
        let err = model_config(
            r#"{"vocab_size": 8, "hidden_size": 24, "num_hidden_layers": 1,
                "num_attention_heads": 6, "num_key_value_heads": 4,
                "intermediate_size": 32}"#,
        )
        .unwrap_err();
        assert!(err.to_string().contains("GQA"), "{err}");
    }

    #[test]
    fn rejects_non_positive_rope_theta() {
        let err = model_config(
            r#"{"vocab_size": 8, "hidden_size": 16, "num_hidden_layers": 1,
                "num_attention_heads": 2, "intermediate_size": 32, "rope_theta": 0}"#,
        )
        .unwrap_err();
        assert!(err.to_string().contains("rope_theta"), "{err}");
    }
}
