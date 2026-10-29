use candle_core::{Device, Tensor};
use llm_core::{ForwardBatch, ForwardContext, models::Model};
use llm_types::{ModelConfig, Result};

use crate::{
    model_executor::model_loader::ModelLoader,
    sampler::{Sampler, SamplerKind, create_sampler},
};

pub struct ModelRunner {
    model: Box<dyn Model<Input = Tensor, Output = Tensor>>,
    sampler: Box<dyn Sampler>,
}

pub struct ModelRunnerOutput {
    /// batch中每个请求下一token对应的logits
    /// shape [batch_size vocab_size]
    pub next_logits: Tensor,
}

impl ModelRunner {
    pub fn init(model_config: &ModelConfig, device: &Device) -> Result<Self> {
        let model = ModelLoader::load(model_config, device)?;
        let sampler = create_sampler(SamplerKind::Greedy);
        Ok(Self { model, sampler })
    }

    pub fn forward(&self, forward_batch: &ForwardBatch) -> Result<ModelRunnerOutput> {
        // forward batch中的tensor都为对应设备上的值
        // 组装forward context 然后调用模型
        let mut temp_attention = String::from("temp");
        let mut forward_context = ForwardContext {
            batch: forward_batch,
            attention: temp_attention.as_mut_str(),
        };
        // 调用self 模型中的forwrad
        // 此处input的数据实际上是input ids
        let logits = self
            .model
            .forward(&forward_batch.input_ids, &mut forward_context)?;
        // output logits中需要根据cur seqlens q来获取next logits
        let last_indices = (&forward_batch.cu_seqlens_q - 1.0)?;
        let next_logits = logits.index_select(&last_indices, 0)?;
        Ok(ModelRunnerOutput { next_logits })
    }

    pub fn sample(&self, logits: &Tensor) -> Result<Tensor> {
        let next_ids = self.sampler.sample(logits)?;
        Ok(next_ids)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use std::time::Instant;

    use llm_types::DeviceSpec;

    use crate::model_executor::weight::index::find_safetensors_files;

    /// 默认模型目录：workspace 上一级的 `models/MiniCPM5-2B`。
    ///
    /// 用 `CARGO_MANIFEST_DIR` 拼绝对路径而不是相对路径——`cargo test` 的工作
    /// 目录是包目录（`crates/llm-runtime`），不是 workspace 根，相对路径会随调用
    /// 方式变化。`CARGO_MANIFEST_DIR` 是 `<workspace>/crates/llm-runtime`，
    /// 往上三级即 `<workspace>` 的父目录。
    ///
    /// 模型放在别处就用环境变量覆盖：
    ///
    /// ```bash
    /// LLM_ENGINE_MODEL_DIR=/path/to/model cargo test ...
    /// ```
    fn model_dir() -> PathBuf {
        match std::env::var("LLM_ENGINE_MODEL_DIR") {
            Ok(path) => PathBuf::from(path),
            Err(_) => PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../models/MiniCPM5-2B"),
        }
    }

    /// 运行设备，默认 CPU。可用 `LLM_ENGINE_DEVICE` 覆盖
    /// （`cpu` / `cuda` / `cuda:1` / `metal`）。
    ///
    /// 默认 CPU 是因为这个测试只验证加载链路、不验证算力；而且 `ModelLoader`
    /// 目前把 `VarBuilder` 的源精度写死成 `DType::F32`，模型构造时再
    /// `set_dtype(BF16)`，在 GPU 上会多做一次 bf16 -> f32 -> bf16 的来回转换。
    fn test_device() -> Result<Device> {
        let spec = std::env::var("LLM_ENGINE_DEVICE").unwrap_or_else(|_| "cpu".to_owned());
        DeviceSpec::parse(&spec)?.build()
    }

    /// 从磁盘加载 MiniCPM5-2B，验证
    /// 「读 config.json -> 解析权重索引 -> 建 VarBuilder -> 构造模型」这条链路。
    ///
    /// 需要真实权重（约 5GB），所以标记 `#[ignore]`，不进默认测试集。
    ///
    /// ```bash
    /// # 默认从 ../models/MiniCPM5-2B 加载，CPU
    /// cargo test -p llm-runtime --release -- --ignored --nocapture loads_minicpm_2b
    ///
    /// # 指定模型目录与设备
    /// LLM_ENGINE_MODEL_DIR=/data/MiniCPM5-2B LLM_ENGINE_DEVICE=cuda:0 \
    ///     cargo test -p llm-runtime --release -- --ignored --nocapture loads_minicpm_2b
    /// ```
    ///
    /// 用 `--release` 是有意的：debug 下加载 42 层、约 2.5B 参数会慢一个数量级。
    ///
    /// 内存：权重常驻一份（bf16，约 5GB），加上 mmap 页缓存，建议留 8GB 以上。
    /// 每个张量在 `vb.get()` 时会先按 `DType::F32` 物化再转 bf16，所以峰值还要
    /// 加上「单个最大张量」的 f32 大小——embedding 是 130560 x 2048，约 1GB。
    #[test]
    #[ignore = "需要 MiniCPM5-2B 权重（约 5GB），不进默认测试集"]
    fn loads_minicpm_2b() -> Result<()> {
        let dir = model_dir();
        println!("模型目录: {}", dir.display());

        let config_path = dir.join("config.json");
        assert!(
            config_path.is_file(),
            "找不到 {}；可用 LLM_ENGINE_MODEL_DIR 指定模型目录",
            config_path.display()
        );

        // --- 1. 配置：读 config.json 并解析出运行所需字段 ---
        let config = ModelConfig::from_model_path(&dir)?;
        println!(
            "架构: {:?}\n\
             层数: {}   hidden: {}   ffn: {}\n\
             heads: {} q / {} kv（每组 {} 个）\n\
             head_dim: {}   vocab: {}   max_pos: {}\n\
             rms_norm_eps: {:e}   rope_theta: {:e}",
            config.arch_name(),
            config.num_hidden_layers,
            config.hidden_size,
            config.intermediate_size,
            config.num_attention_heads,
            config.num_key_value_heads,
            config.num_kv_groups(),
            config.head_dim,
            config.vocab_size,
            config.max_position_embeddings,
            config.rms_norm_eps,
            config.rope_theta,
        );

        // 分发靠这个字段，拿不到就走不到模型构造
        assert_eq!(
            config.arch_name(),
            Some("LlamaForCausalLM"),
            "MiniCPM5-2B 的 architectures 应当是 LlamaForCausalLM"
        );

        // --- 2. 权重索引：解析 model.safetensors.index.json ---
        let shards = find_safetensors_files(&dir)?;
        println!("权重分片（{} 个）:", shards.len());
        for shard in &shards {
            let size = std::fs::metadata(shard).map(|m| m.len()).unwrap_or(0);
            println!("  {} ({:.2} GB)", shard.display(), size as f64 / 1e9);
        }

        assert!(!shards.is_empty(), "权重索引里一个分片都没有");
        for shard in &shards {
            assert!(shard.is_file(), "索引指向的分片不存在: {}", shard.display());
        }

        // --- 3. 加载模型 ---
        let device = test_device()?;
        println!("\n设备: {device:?}，开始加载...");

        let started = Instant::now();
        let _runner = ModelRunner::init(&config, &device)?;
        println!("加载完成，耗时 {:?}", started.elapsed());

        Ok(())
    }
}
