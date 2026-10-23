use crate::forward_batch::ForwardBatch;

pub struct ForwardContext<'a> {
    pub batch: &'a ForwardBatch,
    // temporary use str 占位，目前不使用
    pub attention: &'a mut str,
}
