use candle_core::{D::Minus1, Tensor};
use llm_types::Result;

use super::Sampler;

pub struct GreedySampler;

impl Sampler for GreedySampler {
    fn sample(&self, logits: &Tensor) -> Result<Tensor> {
        let ids = logits.argmax(Minus1)?;
        Ok(ids)
    }
}
