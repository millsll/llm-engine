use super::*;
use llm_tokenizer::{self, hf_tokenizer::HfTokenizer, tokenizer::Tokenizer};

#[test]
fn test_tp_worker_generation_batch_size_1() {
    println!("generation test start---------------");
    let model_config =
        ModelConfig::from_model_path("/home/mills/workspace/models/MiniCPM5-2B").unwrap();
    let device = Arc::new(Device::new_cuda(0).unwrap());
    let tp_worker = TpModelWorker::init(&model_config, device.clone()).unwrap();
    let tokenizer = HfTokenizer::from_file(
        "/home/mills/workspace/models/MiniCPM5-2B/tokenizer.json",
        "/home/mills/workspace/models/MiniCPM5-2B/tokenizer_config.json",
    )
    .unwrap();
    println!("load completed-----------------");
    // 获得一下config以便控制自回归停止
    let tokenizer_config = tokenizer.config();
    //输入一个测试语句，交给tokenizer进行encode
    let mut test_example = "this is a intoduction to Peking University: ".to_string();
    let mut next_token = "".to_string();
    let mut max_tokens = 0usize;
    testlog::test_log!("start deocde-------------");
    while !tokenizer_config.is_eos(&next_token) && max_tokens < 100 {
        let input_ids = tokenizer.encode(&test_example, false).unwrap();
        let len = input_ids.len();
        let next_ids = tp_worker.test_forward(&input_ids).unwrap();
        let next_ids = next_ids.to_vec1().unwrap();
        println!("generate next ids:{:?}", next_ids);
        next_token = tokenizer.decode(&next_ids, false).unwrap();
        println!("generate next token:{next_token}");
        test_example += &next_token;
        println!("next token:{}", next_token);
        max_tokens += 1;
    }
    println!("{test_example}");
}

#[test]
fn test_tp_worker_generation_batch_size_n() {
    println!("generation test start---------------");
    let model_config =
        ModelConfig::from_model_path("/home/mills/workspace/models/MiniCPM5-2B").unwrap();
    let device = Arc::new(Device::new_cuda(0).unwrap());
    let tp_worker = TpModelWorker::init(&model_config, device.clone()).unwrap();
    let tokenizer = HfTokenizer::from_file(
        "/home/mills/workspace/models/MiniCPM5-2B/tokenizer.json",
        "/home/mills/workspace/models/MiniCPM5-2B/tokenizer_config.json",
    )
    .unwrap();
    println!("load completed-----------------");
    let tokenizer_config = tokenizer.config();
    // 编写两个测试样例以测试batch
    let mut test_example1 = "```python\n def fibonacci".to_string();
    let mut test_example2 = "```java\n int bisec".to_string();
    //let mut examples: Vec<String> = vec![test_example1, test_example2];
    let mut next_token1 = "".to_string();
    let mut next_token2 = "".to_string();

    let mut max_tokens = 0usize;
    testlog::test_log!("start deocde-------------");
    while max_tokens < 100 {
        let mut input_ids: Vec<Vec<u32>> = Vec::new();
        if !tokenizer_config.is_eos(&next_token1) {
            let ids1 = tokenizer.encode(&test_example1, false).unwrap();
            input_ids.push(ids1);
        }
        if !tokenizer_config.is_eos(&next_token2) {
            let ids2 = tokenizer.encode(&test_example2, false).unwrap();
            input_ids.push(ids2);
        }
        if input_ids.len() == 0 {
            break;
        }
        let ids_slice: Vec<&[u32]> = input_ids.iter().map(|r| r.as_slice()).collect();
        let next_ids = tp_worker.test_batch_forward(&ids_slice).unwrap();
        let next_ids = next_ids.to_vec1().unwrap();
        next_token1 = tokenizer.decode(&next_ids[0..1], false).unwrap();
        next_token2 = tokenizer.decode(&next_ids[1..], false).unwrap();
        test_example1 += &next_token1;
        test_example2 += &next_token2;
        max_tokens += 1;
    }
    println!("{test_example1}\n");
    println!("{test_example2}\n");
}

#[test]
fn cuda_test() {
    let device = Device::new_cuda(0).unwrap();
    let a = Tensor::ones((1024, 1024), candle_core::DType::F32, &device).unwrap();
    let b = a.matmul(&a).unwrap();
    device.synchronize().unwrap();
}
