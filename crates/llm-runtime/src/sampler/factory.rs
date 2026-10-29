use super::GreedySampler;
use super::Sampler;

#[derive(Debug, Clone, Copy)]
pub enum SamplerKind {
    Greedy,
    Random,
}

pub fn create_sampler(kind: SamplerKind) -> Box<dyn Sampler> {
    match kind {
        SamplerKind::Greedy => Box::new(GreedySampler),
        SamplerKind::Random => Box::new(GreedySampler),
    }
}
