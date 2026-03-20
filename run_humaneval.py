"""
任务1: HumanEval 代码补全评测
数据集: openai/openai_humaneval (164道Python编程题)
指标: pass@k

  python tasks/task1_code_completion/run_humaneval.py \
      --model codellama \
      --num_samples 1 \
      --output_dir /home/xyl/code_llm_survey/results/task1
"""

import os
import sys
import json
import argparse
import logging
import time
import threading
import re
from pathlib import Path
from datetime import datetime
from math import comb

import torch
import numpy as np
from datasets import load_dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from models.model_loader import load_model, build_prompt, generate, extract_code_from_response

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# =================== 提示词 ===================
HUMANEVAL_SYSTEM_PROMPT = """You are an expert Python programmer.
Your task is to complete the given Python function.
- Output ONLY the function body (lines that go inside the function), properly indented with 4 spaces
- Do NOT repeat the function signature or docstring
- Do NOT include any explanation, markdown, or code fences
- Output valid Python code only"""


def format_humaneval_prompt(problem: dict, model_name: str) -> str:
    """格式化HumanEval提示词"""
    prompt_text = problem["prompt"]
    instruction = (
        f"Complete the following Python function. "
        f"Only provide the implementation body, not the signature:\n\n"
        f"{prompt_text}"
    )
    return build_prompt(model_name, instruction, system_prompt=HUMANEVAL_SYSTEM_PROMPT)


def clean_completion(raw: str, prompt: str) -> str:
    """清理模型输出，确保格式正确"""
    raw = re.sub(r"```(?:python)?\n?", "", raw)
    raw = re.sub(r"```\s*$", "", raw, flags=re.MULTILINE)
    raw = raw.strip()

    lines = raw.split('\n')
    result_lines = []
    skip_mode = False
    for line in lines:
        stripped = line.strip()
        if re.match(r'^def\s+', stripped):
            func_name_match = re.search(r'^def\s+(\w+)', stripped)
            if func_name_match and func_name_match.group(1) in prompt:
                skip_mode = True
                continue
        if skip_mode:
            if stripped.startswith('"""') or stripped.startswith("'''"):
                # 等待docstring结束
                quote = '"""' if stripped.startswith('"""') else "'''"
                if stripped.count(quote) >= 2 or (len(stripped) > 3 and stripped.endswith(quote)):
                    skip_mode = False
                continue
            skip_mode = False
        result_lines.append(line)

    result = '\n'.join(result_lines).strip()

    if not result:
        return "    pass"

    # 确保有4空格缩进
    if result and not result.startswith('    ') and not result.startswith('\t'):
        result = '\n'.join(
            '    ' + line if line.strip() else line
            for line in result.split('\n')
        )
    return result


def _exec_worker_he(code, q):
    try:
        exec(compile(code, "<string>", "exec"), {})
        q.put({"passed": True, "error": None})
    except AssertionError as e:
        q.put({"passed": False, "error": f"AssertionError: {e}"})
    except Exception as e:
        q.put({"passed": False, "error": f"{type(e).__name__}: {e}"})


def run_code_with_timeout(code: str, timeout: int = 8) -> dict:
    import multiprocessing as mp
    ctx = mp.get_context("fork")
    q = ctx.Queue()
    p = ctx.Process(target=_exec_worker_he, args=(code, q), daemon=True)
    p.start()
    p.join(timeout=timeout)
    if p.is_alive():
        p.terminate(); p.join(timeout=2)
        if p.is_alive(): p.kill(); p.join()
        return {"passed": False, "error": f"Timeout (>{timeout}s)"}
    return q.get() if not q.empty() else {"passed": False, "error": "No result"}


def evaluate_completion(problem: dict, completion: str) -> dict:
    full_code = (
        problem["prompt"]
        + completion
        + "\n\n"
        + problem["test"]
        + f"\n\ncheck({problem['entry_point']})\n"
    )
    return run_code_with_timeout(full_code, timeout=10)


def compute_pass_at_k(n: int, c: int, k: int) -> float:
    if n < k:
        return 0.0
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


def load_humaneval_dataset():
    """加载HumanEval"""
    candidates = [
        ("openai/openai_humaneval", "test"),
        ("openai_humaneval", "test"),
    ]
    for name, split in candidates:
        try:
            ds = load_dataset(name, split=split, trust_remote_code=True)
            logger.info(f"数据集加载成功: {name} ({len(ds)} 题)")
            return list(ds)
        except Exception as e:
            logger.warning(f"  {name} 失败: {e}")

    # 备用：直接下载JSON
    logger.warning("备用：直接下载JSON的HumanEval")
    try:
        import gzip, urllib.request
        url = "https://raw.githubusercontent.com/openai/human-eval/master/data/HumanEval.jsonl.gz"
        with urllib.request.urlopen(url, timeout=30) as resp:
            content = gzip.decompress(resp.read()).decode('utf-8')
        problems = [json.loads(l) for l in content.strip().split('\n') if l.strip()]
        logger.info(f"备用成功: {len(problems)} 题")
        return problems
    except Exception as e:
        raise RuntimeError(f"HumanEval数据集加载失败，换！: {e}")


def run_humaneval(
    model_name: str,
    num_samples: int = 1,
    temperature: float = 0.2,
    max_new_tokens: int = 512,
    use_4bit: bool = False,
    output_dir: str = "/home/xyl/code_llm_survey/results/task1",
    max_problems: int = None,
    cache_dir: str = None,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- 加载模型 ---
    logger.info(f"加载模型: {model_name}")
    model, tokenizer, config = load_model(
        model_name, use_4bit=use_4bit, cache_dir=cache_dir
    )

    # --- 加载数据集 ---
    problems = load_humaneval_dataset()
    if max_problems:
        problems = problems[:max_problems]
    logger.info(f"评测 {len(problems)} 道题，每题生成 {num_samples} 次")

    # --- 评测 ---
    all_results = []
    stats_list = []
    error_types = {}
    total_passed = 0
    total_gen = 0
    start_time = time.time()

    for i, problem in enumerate(tqdm(problems, desc=f"HumanEval [{config['display_name']}]")):
        task_id = problem["task_id"]
        n, c = 0, 0
        comps = []

        for s in range(num_samples):
            try:
                prompt = format_humaneval_prompt(problem, model_name)
                do_sample = (num_samples > 1) and (temperature > 0)
                outputs = generate(
                    model, tokenizer, prompt, config,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=do_sample,
                )
                raw = outputs[0]
                completion = clean_completion(raw, problem["prompt"])
                eval_r = evaluate_completion(problem, completion)

                n += 1; total_gen += 1
                if eval_r["passed"]:
                    c += 1; total_passed += 1
                else:
                    ek = (eval_r["error"] or "Unknown").split(":")[0]
                    error_types[ek] = error_types.get(ek, 0) + 1

                comps.append({
                    "sample": s,
                    "raw_output": raw[:800],
                    "completion": completion[:800],
                    "passed": eval_r["passed"],
                    "error": eval_r["error"],
                })
            except Exception as e:
                logger.error(f"  {task_id} s{s}: {e}")
                n += 1; total_gen += 1

        stats_list.append({"n": n, "c": c})
        all_results.append({
            "task_id": task_id,
            "prompt": problem["prompt"],
            "canonical_solution": problem.get("canonical_solution", ""),
            "completions": comps,
            "n": n, "c": c,
        })

        # 进度日志（每10题）
        if (i + 1) % 10 == 0 or (i + 1) == len(problems):
            cur_p1 = total_passed / total_gen if total_gen else 0
            elapsed = time.time() - start_time
            eta = elapsed / (i + 1) * (len(problems) - i - 1)
            logger.info(
                f"  [{i+1}/{len(problems)}] 即时pass@1={cur_p1:.3f} | "
                f"已用{elapsed/60:.1f}min | 剩余约{eta/60:.1f}min"
            )

    elapsed = time.time() - start_time

    # --- 计算 pass@k ---
    pass_k = {}
    for k in [1, 5, 10]:
        vals = [compute_pass_at_k(s["n"], s["c"], k)
                for s in stats_list if s["n"] >= k]
        if vals:
            pass_k[f"pass@{k}"] = float(np.mean(vals))

    metrics = {
        "model": model_name,
        "display_name": config["display_name"],
        "dataset": "HumanEval",
        "total_problems": len(problems),
        "num_samples": num_samples,
        "temperature": temperature,
        **pass_k,
        "passed_total": total_passed,
        "total_generated": total_gen,
        "elapsed_seconds": round(elapsed, 1),
        "avg_sec_per_problem": round(elapsed / max(len(problems), 1), 2),
        "error_distribution": error_types,
        "timestamp": datetime.now().isoformat(),
    }

    # --- 保存 ---
    out_file = output_dir / f"humaneval_{model_name}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({"metrics": metrics, "results": all_results},
                  f, indent=2, ensure_ascii=False)

    # --- 汇总 ---
    sep = "=" * 62
    logger.info(f"\n{sep}")
    logger.info(f"  模型:    {config['display_name']}")
    logger.info(f"  数据集:  HumanEval ({len(problems)} 题)")
    for k_str, v in pass_k.items():
        logger.info(f"  {k_str}:  {v:.4f}")
    logger.info(f"  耗时:    {elapsed/60:.1f} 分钟")
    logger.info(f"  错误:    {error_types}")
    logger.info(f"  输出:    {out_file}")
    logger.info(sep)
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["codellama", "qwen2coder"], required=True)
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--use_4bit", action="store_true")
    parser.add_argument("--output_dir",
                        default="/home/xyl/code_llm_survey/results/task1")
    parser.add_argument("--max_problems", type=int, default=None)
    parser.add_argument("--cache_dir", default=None)
    args = parser.parse_args()

    m = run_humaneval(
        model_name=args.model,
        num_samples=args.num_samples,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        use_4bit=args.use_4bit,
        output_dir=args.output_dir,
        max_problems=args.max_problems,
        cache_dir=args.cache_dir,
    )
    print(f"\n HumanEval pass@1 = {m.get('pass@1', 'N/A'):.4f}")
