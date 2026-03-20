"""
CodeLlama-7B-Instruct 和 Qwen2.5-Coder-7B-Instruct
"""

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    pipeline
)
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =================== 模型配置 ===================
MODEL_CONFIGS = {
    "codellama": {
        "model_id": "codellama/CodeLlama-7b-Instruct-hf",
        "display_name": "CodeLlama-7B-Instruct",
        "context_length": 16384,
        "prompt_template": "instruct",
        "stop_tokens": ["[/INST]", "</s>"],
        "temperature": 0.2,
        "top_p": 0.95,
    },
    "qwen2coder": {
        "model_id": "Qwen/Qwen2.5-Coder-7B-Instruct",
        "display_name": "Qwen2.5-Coder-7B-Instruct",
        "context_length": 32768,
        "prompt_template": "chatml",
        "stop_tokens": ["<|im_end|>", "<|endoftext|>"],
        "temperature": 0.2,
        "top_p": 0.95,
    },
}


def get_quantization_config(use_4bit=False, use_8bit=False):
    if use_4bit:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    elif use_8bit:
        return BitsAndBytesConfig(load_in_8bit=True)
    return None


def load_model(
    model_name: str,
    use_4bit: bool = False,
    use_8bit: bool = False,
    device_map: str = "auto",
    cache_dir: str = None,
):
    """
    加载指定模型和分词器
    """
    assert model_name in MODEL_CONFIGS, f"未知模型: {model_name}，可选: {list(MODEL_CONFIGS.keys())}"

    config = MODEL_CONFIGS[model_name]
    model_id = config["model_id"]

    logger.info(f"加载模型: {config['display_name']}")
    logger.info(f"  模型ID: {model_id}")
    logger.info(f"  量化: {'4bit' if use_4bit else '8bit' if use_8bit else 'FP16'}")

    if torch.cuda.is_available():
        free_gb = torch.cuda.mem_get_info()[0] / 1024**3
        logger.info(f"  可用显存: {free_gb:.1f}GB")
        if free_gb < 14 and not (use_4bit or use_8bit):
            logger.warning("显存不足")

    # 加载分词器
    import os as _os
    _os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    _os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

    start = time.time()
    # 优先本地缓存
    def _load_tokenizer(model_id, cache_dir, **kwargs):
        try:
            tok = AutoTokenizer.from_pretrained(
                model_id, local_files_only=True,
                cache_dir=cache_dir, padding_side="left", **kwargs
            )
            logger.info("  Tokenizer: 从本地缓存加载")
            return tok
        except Exception:
            pass
        # 在线下载
        logger.info("  Tokenizer: 从网络加载")
        return AutoTokenizer.from_pretrained(
            model_id, local_files_only=False,
            cache_dir=cache_dir, padding_side="left", **kwargs
        )

    tokenizer = _load_tokenizer(model_id, cache_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 加载模型
    quant_config = get_quantization_config(use_4bit, use_8bit)
    dtype = torch.float16 if not (use_4bit or use_8bit) else None

    def _load_model(model_id, **kwargs):
        # 离线加载
        try:
            m = AutoModelForCausalLM.from_pretrained(
                model_id, local_files_only=True, **kwargs)
            logger.info("  Model: 从本地缓存加载")
            return m
        except Exception:
            pass
        logger.info("  Model: 从网络加载")
        return AutoModelForCausalLM.from_pretrained(model_id, local_files_only=False, **kwargs)

    model = _load_model(
        model_id,
        dtype=dtype,
        device_map=device_map,
        quantization_config=quant_config,
        trust_remote_code=True,
        cache_dir=cache_dir,
    )
    model.eval()

    elapsed = time.time() - start
    logger.info(f"  加载完成，耗时: {elapsed:.1f}秒")

    # 打印模型显存占用
    if torch.cuda.is_available():
        mem_used = torch.cuda.memory_allocated() / 1024**3
        logger.info(f"  当前显存占用: {mem_used:.1f}GB")

    return model, tokenizer, config


def build_prompt(
    model_name: str,
    instruction: str,
    system_prompt: str = None,
    config: dict = None,
) -> str:
    """
    提示词：

    CodeLlama格式:
        <s>[INST] <<SYS>>\n{system}\n<</SYS>>\n\n{instruction} [/INST]

    Qwen2.5格式:
        <|im_start|>system\n{system}<|im_end|>\n
        <|im_start|>user\n{instruction}<|im_end|>\n
        <|im_start|>assistant\n
    """
    if config is None:
        config = MODEL_CONFIGS[model_name]

    template = config["prompt_template"]

    if template == "instruct":  # CodeLlama
        if system_prompt:
            prompt = f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{instruction} [/INST]"
        else:
            prompt = f"<s>[INST] {instruction} [/INST]"

    elif template == "chatml":  # Qwen2.5
        parts = []
        if system_prompt:
            parts.append(f"<|im_start|>system\n{system_prompt}<|im_end|>")
        parts.append(f"<|im_start|>user\n{instruction}<|im_end|>")
        parts.append("<|im_start|>assistant\n")
        prompt = "\n".join(parts)

    else:
        if system_prompt:
            prompt = f"{system_prompt}\n\n{instruction}\n"
        else:
            prompt = f"{instruction}\n"

    return prompt


@torch.no_grad()
def generate(
    model,
    tokenizer,
    prompt: str,
    config: dict,
    max_new_tokens: int = 512,
    temperature: float = None,
    top_p: float = None,
    num_return_sequences: int = 1,
    do_sample: bool = True,
) -> list:
    """
    执行生成，返回生成的文本列表（已去除输入部分）
    """
    if temperature is None:
        temperature = config.get("temperature", 0.2)
    if top_p is None:
        top_p = config.get("top_p", 0.95)

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=config.get("context_length", 4096) - max_new_tokens,
    ).to(model.device)

    input_len = inputs["input_ids"].shape[1]

    if temperature <= 0 or not do_sample:
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    else:
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            num_return_sequences=num_return_sequences,
            pad_token_id=tokenizer.pad_token_id,
        )

    results = []
    for output in outputs:
        new_tokens = output[input_len:]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        for stop in config.get("stop_tokens", []):
            if stop in text:
                text = text[:text.index(stop)]
        results.append(text.strip())

    return results


def extract_code_from_response(response: str, language: str = "python") -> str:
    """
    提取代码块
    处理 ```python ... ``` 或直接代码格式
    """
    import re

    patterns = [
        rf"```{language}\n(.*?)```",
        r"```\n(.*?)```",
        r"```(.*?)```",
    ]
    for pattern in patterns:
        match = re.search(pattern, response, re.DOTALL)
        if match:
            return match.group(1).strip()

    return response.strip()


# 测试
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["codellama", "qwen2coder"], required=True)
    parser.add_argument("--use_4bit", action="store_true")
    parser.add_argument("--test_prompt", default="Write a Python function to compute factorial recursively.")
    args = parser.parse_args()

    model, tokenizer, config = load_model(args.model, use_4bit=args.use_4bit)

    prompt = build_prompt(
        args.model,
        args.test_prompt,
        system_prompt="You are an expert programmer. Provide clean, correct code.",
        config=config,
    )
    print(f"\n{'='*50}")
    print(f"测试提示词:\n{prompt}")
    print(f"{'='*50}")

    results = generate(model, tokenizer, prompt, config, max_new_tokens=256)
    print(f"\n生成结果:\n{results[0]}")

    print(f"\n提取的代码:\n{extract_code_from_response(results[0])}")
