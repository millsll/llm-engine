//! # llm-tokenizer
//!
//! 封装 HuggingFace `tokenizers`，对外提供：
//!
//! - `Tokenizer::from_dir(model_dir)`：读取权重目录下的 `tokenizer.json`。
//! - `encode(text, add_special_tokens) -> Vec<u32>`。
//! - `IncrementalDecoder`：流式解码。
//!
//! ## 为什么需要增量解码
//!
//! 流式输出时每个 token 单独解码。一个汉字通常占 3 个字节，可能横跨两个 token
//! 的交界处，直接逐 token 解码会吐出 U+FFFD 替换字符。`IncrementalDecoder`
//! 缓冲尾部不完整的 UTF-8 字节序列，只输出能构成合法字符的部分。
//! 这是 Phase 6 流式接口正确性的前提，Phase 1 的端到端 demo 也会用到。
//!
//! TODO(phase-1): 实现 `Tokenizer` 的加载与编解码。
//! TODO(phase-6): 实现 `IncrementalDecoder` 与 stop-string 匹配。
