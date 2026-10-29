use candle_core::Tensor;
use llm_types::Result;

pub mod factory;
pub mod greedy;

pub use factory::SamplerKind;
pub use factory::create_sampler;
pub use greedy::GreedySampler;

pub trait Sampler {
    fn sample(&self, logits: &Tensor) -> Result<Tensor>;
}
