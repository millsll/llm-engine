use llm_types::Error;
use llm_types::Result;
use std::path::Path;

use tokenizers::Tokenizer;

use crate::hf_tokenizer_config::HfTokenizerConfig;
use crate::tokenizer;

pub struct HfTokenizer {
    inner: Tokenizer,
    tokenizer_config: HfTokenizerConfig,
}

impl HfTokenizer {
    pub fn from_file(
        tokenizer_path: impl AsRef<Path>,
        config_path: impl AsRef<Path>,
    ) -> Result<Self> {
        let inner =
            tokenizers::Tokenizer::from_file(tokenizer_path.as_ref()).map_err(|source| {
                Error::TokenizerLoad {
                    path: tokenizer_path.as_ref().to_path_buf(),
                    source,
                }
            })?;
        let config = HfTokenizerConfig::from_file(config_path)?;
        Ok(Self {
            inner,
            tokenizer_config: config,
        })
    }

    /// `tokenizer_config.json` 里的配置（bos / eos / unk / pad 等）。
    ///
    /// 注意这里读到的 `add_bos_token` 对 fast tokenizer **不生效**：真正决定
    /// 是否加 BOS 的是 `tokenizer.json` 里的 `post_processor`。
    pub fn config(&self) -> &HfTokenizerConfig {
        &self.tokenizer_config
    }
}

impl tokenizer::Tokenizer for HfTokenizer {
    fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>> {
        let encoding = self.inner.encode(text, add_special_tokens)?;
        Ok(encoding.get_ids().to_vec())
    }

    fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String> {
        let decoding_res = self.inner.decode(ids, skip_special_tokens)?;
        Ok(decoding_res)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    // 父模块只 `use crate::tokenizer;`（模块），trait 本身不在作用域里。
    // 用 `as _` 引入以支持方法调用，同时避免和 `tokenizers::Tokenizer` 重名。
    use crate::tokenizer::Tokenizer as _;

    /// 用于 forward 测试的固定输入。中文标点能顺带验证多字节字符
    /// 不会被 token 边界切断。
    const PROMPT: &str = "你好，介绍一下你自己";

    /// 默认模型目录：workspace 上一级的 `models/MiniCPM5-2B`。
    ///
    /// `CARGO_MANIFEST_DIR` 是 `<workspace>/crates/llm-tokenizer`，
    /// 往上三级即 `<workspace>` 的父目录。用绝对路径是因为 `cargo test` 的
    /// 工作目录是包目录而不是 workspace 根。
    ///
    /// 模型放在别处就用 `LLM_ENGINE_MODEL_DIR` 覆盖。
    fn model_dir() -> PathBuf {
        match std::env::var("LLM_ENGINE_MODEL_DIR") {
            Ok(path) => PathBuf::from(path),
            Err(_) => PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../models/MiniCPM5-2B"),
        }
    }

    /// 产物路径：`crates/llm-tokenizer/tmp/greeting_token_ids.json`。
    ///
    /// 放在 crate 目录下而不是系统临时目录，是为了让 forward 测试能按固定
    /// 相对路径找到它。
    fn output_path() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tmp/greeting_token_ids.json")
    }

    /// 读 MiniCPM5-2B 的 tokenizer，编码一句中文问候，把结果落盘。
    ///
    /// 产物给后续 forward 测试当固定输入用，所以两种模式都记下来：
    /// 带特殊 token 的那组对应 HF `tokenizer(text)`，
    /// 不带的那组对应 `tokenizer(text, add_special_tokens=False)`。
    ///
    /// 依赖外部模型目录（`tokenizer.json` 约 10MB），所以标记 `#[ignore]`，
    /// 不进默认测试集。
    ///
    /// ```bash
    /// cargo test -p llm-tokenizer -- --ignored --nocapture encodes_greeting
    /// ```
    ///
    /// 换模型目录：
    ///
    /// ```bash
    /// LLM_ENGINE_MODEL_DIR=/path/to/model \
    ///     cargo test -p llm-tokenizer -- --ignored --nocapture encodes_greeting
    /// ```
    #[test]
    #[ignore = "依赖外部模型目录（MiniCPM5-2B/tokenizer.json），不进默认测试集"]
    fn encodes_greeting_and_writes_ids() -> Result<()> {
        let dir = model_dir();
        let tokenizer_path = dir.join("tokenizer.json");
        let config_path = dir.join("tokenizer_config.json");

        assert!(
            tokenizer_path.is_file(),
            "找不到 {}；可用 LLM_ENGINE_MODEL_DIR 指定模型目录",
            tokenizer_path.display()
        );
        assert!(
            config_path.is_file(),
            "找不到 {}；可用 LLM_ENGINE_MODEL_DIR 指定模型目录",
            config_path.display()
        );

        let started = std::time::Instant::now();
        let hf = HfTokenizer::from_file(&tokenizer_path, &config_path)?;
        println!("tokenizer 加载耗时 {:?}", started.elapsed());
        println!(
            "config: bos={:?} eos={:?} pad={:?}",
            hf.config().bos_token,
            hf.config().eos_token,
            hf.config().pad_token
        );

        // 主用例：与 HF `tokenizer(text)` 一致，走 tokenizer.json 里的 post_processor
        let ids = hf.encode(PROMPT, true)?;
        let ids_no_special = hf.encode(PROMPT, false)?;
        let decoded = hf.decode(&ids, true)?;

        println!("prompt : {PROMPT}");
        println!("ids    : {ids:?}");
        println!("无特殊 : {ids_no_special:?}");
        println!("decoded: {decoded}");

        assert!(!ids.is_empty(), "编码结果为空");
        assert!(!decoded.is_empty(), "解码结果为空");
        // 中文必须原样还原；这里失败说明 tokenizer.json 与预期模型不匹配
        assert_eq!(decoded, PROMPT, "解码回原文失败");

        let payload = serde_json::json!({
            "model_dir": dir.display().to_string(),
            "tokenizer_file": tokenizer_path.display().to_string(),
            "prompt": PROMPT,
            "add_special_tokens": true,
            "token_ids": ids,
            "num_tokens": ids.len(),
            "decoded": decoded,
            "token_ids_no_special": ids_no_special,
            "num_tokens_no_special": ids_no_special.len(),
        });

        let out = output_path();
        if let Some(parent) = out.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(&out, serde_json::to_string_pretty(&payload)?)?;
        println!("已写入: {}", out.display());

        Ok(())
    }
}
