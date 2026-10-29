use std::path::PathBuf;

#[derive(Debug, Clone)]
pub struct ServerArgs {
    /// 本地模型路径
    pub model_path: PathBuf,
    /// 可选配置文件路径
    pub model_config_path: Option<PathBuf>,
}
