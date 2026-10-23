use candle_core::DType;
use candle_core::Device;
use candle_core::Tensor;
use candle_nn::VarBuilder;
use llm_core::models::{Model, llama::LlamaForCausalLM};
use llm_types::Error;
use llm_types::ModelConfig;
use llm_types::Result;
use llm_types::model_config::LlamaConfig;

use crate::model_executor::weight::index::find_safetensors_files;

pub struct ModelLoader;

impl ModelLoader {
    pub fn load(
        config: &ModelConfig,
        device: &Device,
    ) -> Result<Box<dyn Model<Input = Tensor, Output = Tensor>>> {
        // 根据path创建vb
        let model_dir = config.model_path();
        let shards = find_safetensors_files(model_dir)?;

        #[allow(unsafe_code)]
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&shards, DType::F32, device)? };

        // 根据arch派发具体type
        let arch: String;
        match config.arch_name() {
            Some(name) => {
                arch = name.to_string();
            }
            None => return Err(Error::Config("no model arch".to_string())),
        }

        //  转换到对应的model config然后传入load model
        let model: Box<dyn Model<Input = Tensor, Output = Tensor>>;
        match arch.as_str() {
            "LlamaForCausalLM" => {
                // config转换
                model = Box::new(LlamaForCausalLM::load(
                    &vb,
                    &LlamaConfig::from_model_config(config)?,
                )?)
            }
            _ => return Err(Error::Config("model arch not support".to_string())),
        };
        Ok(model)
    }
}
