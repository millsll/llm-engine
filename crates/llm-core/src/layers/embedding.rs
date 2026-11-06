use candle_core::{DType, Tensor};
use candle_nn::VarBuilder;
use llm_types::Result;

use crate::Module;
pub struct VocabeEmbedding {
    weight: Tensor,
}

impl Module for VocabeEmbedding {
    type Input = Tensor;
    type Output = Tensor;
    fn forward(&self, input: &Tensor, _ctx: &mut crate::ForwardContext) -> Result<Tensor> {
        assert!(input.dtype() == DType::U32);
        testlog::test_log!(
            "weight dtype:{:?}, input dtype:{:?}",
            self.weight.dtype(),
            input.dtype()
        );
        let max_index = input.max_all()?.to_scalar::<u32>()?;
        if max_index as usize >= self.weight.dim(0)? {
            return Err(llm_types::Error::InvalidRequest(
                "index select out of bound in embedding".to_string(),
            ));
        }
        let output = self.weight.index_select(input, 0)?;
        Ok(output)
    }
}

impl VocabeEmbedding {
    pub fn load(vb: &VarBuilder, vocabe_shape: &[usize]) -> Result<Self> {
        let weight = vb.get(vocabe_shape, "weight")?;
        Ok(Self { weight })
    }
}
