use crate::forward_batch::ForwardBatch;

pub struct ForwardContext<'a> {
    pub batch: &'a ForwardBatch,
    // temporary use str 占位
    pub attention: &'a mut str,
}
