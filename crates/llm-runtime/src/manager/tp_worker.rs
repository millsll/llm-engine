use crate::model_executor::model_runner::ModelRunner;
use candle_core::{Device, Tensor};
use llm_core::{ForwardBatch, forward_batch};
use llm_types::{ModelConfig, Result};
use std::sync::Arc;

pub struct TpModelWorker {
    model_runner: ModelRunner,
    device: Arc<Device>,
}

impl TpModelWorker {
    pub fn init(model_config: &ModelConfig, device: Arc<Device>) -> Result<Self> {
        // 初始化内部runner
        let model_runner = ModelRunner::init(model_config, &device)?;
        Ok(Self {
            model_runner,
            device,
        })
    }

    pub fn forward_batch_generation() {}

    /// 测试接口，直接传入一个batch size 为1 的内容，交给内部持有的runner进行运行
    /// 对得到的结果进行sample 得到next tokenlogits
    pub fn test_forward(&self, input_ids: &[u32]) -> Result<Tensor> {
        // 输入一个tokenids tensor
        // 交给内部runner执行，获得sample后的id交给tokenizer进行detokenize

        // 创建forward batch，size 为1，直接一维展开
        let len = input_ids.len();
        let positions: Vec<u32> = (0u32..len as u32).collect();
        let query_lens = vec![len];
        let prefix_lens = vec![0usize];
        let forward_batch = ForwardBatch::new(
            input_ids,
            &positions,
            &query_lens,
            &prefix_lens,
            &self.device,
        )?;

        //let forward_batch = ForwardBatch::new(input_ids, &positions, &query_lens, &prefix_lens)?;
        let model_output = self.model_runner.forward(&forward_batch)?;
        let logits = model_output.next_logits;
        // sample一下得到nextids
        let next_ids = self.model_runner.sample(&logits)?;
        Ok(next_ids)
    }

    pub fn test_batch_forward(&self, input_ids: &[&[u32]]) -> Result<Tensor> {
        let forward_batch = ForwardBatch::prefill_from_input_ids(input_ids, &self.device)?;
        let model_output = self.model_runner.forward(&forward_batch)?;
        let logits = model_output.next_logits;
        let next_ids = self.model_runner.sample(&logits)?;
        Ok(next_ids)
    }
}

#[cfg(test)]
#[path = "tests.rs"]
mod tests;
