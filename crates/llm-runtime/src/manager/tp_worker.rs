use crate::model_executor::model_runner::ModelRunner;
use llm_types::Result;

pub struct TpModelWorker {
    model_runner: ModelRunner,
}

impl TpModelWorker {
    pub fn init() -> Result<Self> {
        Ok(())
    }

    pub fn forward_batch_generation() {}
}
