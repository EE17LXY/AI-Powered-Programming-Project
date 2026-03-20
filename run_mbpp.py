"""
任务1: MBPP 代码补全评测

  python tasks/task1_code_completion/run_mbpp.py \
      --model codellama --output_dir results/task1
"""

import os
import sys
import json
import argparse
import logging
import time
import textwrap
import re
import multiprocessing as mp
from pathlib import Path
from datetime import datetime
from math import comb

import torch
import numpy as np
from datasets import load_dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from models.model_loader import load_model, build_prompt, generate, extract_code_from_response

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


MBPP_SYSTEM = """You are an expert Python programmer.
Write a complete, standalone Python function that solves the given problem.
- Include the full function definition (def ...)
- Use correct Python indentation (4 spaces)
- Output ONLY the Python function, no explanations, no markdown fences
- Make sure the function name matches what the tests expect"""


def format_mbpp_prompt(problem: dict, model_name: str) -> str:
    tests = problem.get("test_list", [])
    test_str = "\n".join(tests[:3])
    instruction = (
        f"Write a Python function to solve the following task.\n\n"
        f"Task: {problem['text']}\n\n"
        f"Your function must pass these tests:\n{test_str}\n\n"
        f"Write only the complete Python function:"
    )
    return build_prompt(model_name, instruction, system_prompt=MBPP_SYSTEM)


def extract_and_fix_code(raw: str) -> str:
    """提取修复Python代码，处理缩进"""
    fence = re.search(r"```(?:python)?\s*\n(.*?)```", raw, re.DOTALL)
    if fence:
        code = fence.group(1).strip()
    else:
        # 找第一个def
        def_m = re.search(r'^def\s+\w+', raw, re.MULTILINE)
        if def_m:
            code = raw[def_m.start():].strip()
            # 截断到下一个顶级定义
            lines = code.split('\n')
            result = [lines[0]]
            for line in lines[1:]:
                if line.strip() == '':
                    result.append(line)
                elif line.startswith((' ', '\t')):
                    result.append(line)
                elif re.match(r'^(def |class |import |from |if __name__)', line):
                    break
                else:
                    result.append(line)
            code = '\n'.join(result).strip()
        else:
            code = raw.strip()

    try:
        code = textwrap.dedent(code)
    except Exception:
        pass

    # 修复缩进
    lines = code.split('\n')
    if (len(lines) >= 2 and re.match(r'^def\s+', lines[0]) and
            lines[1].strip() and
            not lines[1].startswith((' ', '\t')) and
            not lines[1].strip().startswith('#')):
        fixed = [lines[0]]
        for line in lines[1:]:
            fixed.append('    ' + line if line.strip() else line)
        code = '\n'.join(fixed)

    return code


def _exec_worker(code: str, result_queue: mp.Queue):
    try:
        exec(compile(code, "<string>", "exec"), {})
        result_queue.put({"passed": True, "error": None})
    except AssertionError as e:
        result_queue.put({"passed": False, "error": f"AssertionError: {e}"})
    except Exception as e:
        result_queue.put({"passed": False, "error": f"{type(e).__name__}: {e}"})


def run_code_safe(code: str, timeout: int = 8) -> dict:
    # 对于 fork 安全性，使用 spawn 上下文
    ctx = mp.get_context("fork")  # fork更快，Linux上安全
    q = ctx.Queue()
    p = ctx.Process(target=_exec_worker, args=(code, q), daemon=True)
    p.start()
    p.join(timeout=timeout)

    if p.is_alive():
        p.terminate()
        p.join(timeout=2)
        if p.is_alive():
            p.kill()
            p.join()
        return {"passed": False, "error": f"Timeout (>{timeout}s)"}

    if not q.empty():
        return q.get()
    return {"passed": False, "error": "No result (process crashed)"}


def evaluate_mbpp_code(problem: dict, code: str) -> dict:
    test_code = "\n".join(problem.get("test_list", []))
    full_code = code + "\n\n" + test_code
    return run_code_safe(full_code, timeout=8)


def compute_pass_at_k(n: int, c: int, k: int) -> float:
    if n < k: return 0.0
    if n - c < k: return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


def run_mbpp(
    model_name: str,
    num_samples: int = 1,
    temperature: float = 0.2,
    max_new_tokens: int = 512,
    use_4bit: bool = False,
    output_dir: str = "/home/xyl/code_llm_survey/results/task1",
    split: str = "test",
    max_problems: int = None,
    cache_dir: str = None,
    resume: bool = True,        # 断点续跑
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- 加载模型 ---
    logger.info(f"加载模型: {model_name}")
    model, tokenizer, config = load_model(
        model_name, use_4bit=use_4bit, cache_dir=cache_dir
    )

    # --- 加载数据集 ---
    logger.info("加载 MBPP 数据集...")
    dataset = load_dataset("google-research-datasets/mbpp", "full", split=split)
    problems = list(dataset)
    if max_problems:
        problems = problems[:max_problems]
    logger.info(f"评测 {len(problems)} 道题，每题生成 {num_samples} 次")

    # --- 断点续跑：加载已有结果 ---
    out_file = output_dir / f"mbpp_{model_name}.json"
    completed_ids = set()
    existing_results = []

    if resume and out_file.exists():
        try:
            with open(out_file) as f:
                old_data = json.load(f)
            existing_results = old_data.get("results", [])
            completed_ids = {r["task_id"] for r in existing_results}
            logger.info(f"断点续跑: 已完成 {len(completed_ids)} 题，继续剩余题目")
        except Exception as e:
            logger.warning(f"读取已有结果失败: {e}，从头开始")
            existing_results = []
            completed_ids = set()

    # --- 评测 ---
    all_results = list(existing_results)
    n_list = [r.get("n", 1) for r in existing_results]
    c_list = [r.get("c", 0) for r in existing_results]
    error_types = {}
    total_passed = sum(c_list)
    total_gen = sum(n_list)
    start_time = time.time()

    remaining = [p for p in problems if p.get("task_id", -1) not in completed_ids]
    logger.info(f"剩余未完成: {len(remaining)} 题")

    for i, problem in enumerate(tqdm(remaining, desc=f"MBPP [{config['display_name']}]")):
        task_id = problem.get("task_id", i)
        n = c = 0
        comps = []

        for s in range(num_samples):
            gen_start = time.time()
            try:
                prompt = format_mbpp_prompt(problem, model_name)
                do_sample = (num_samples > 1) and (temperature > 0)
                outputs = generate(
                    model, tokenizer, prompt, config,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=do_sample,
                )
                raw = outputs[0]
                code = extract_and_fix_code(raw)
                eval_r = evaluate_mbpp_code(problem, code)

                n += 1; total_gen += 1
                if eval_r["passed"]:
                    c += 1; total_passed += 1
                else:
                    ek = (eval_r["error"] or "Unknown").split(":")[0]
                    error_types[ek] = error_types.get(ek, 0) + 1

                gen_time = time.time() - gen_start
                comps.append({
                    "sample": s,
                    "raw_output": raw[:600],
                    "extracted_code": code[:600],
                    "passed": eval_r["passed"],
                    "error": eval_r["error"],
                    "time_sec": round(gen_time, 1),
                })

            except Exception as e:
                logger.error(f"  task {task_id}: {e}")
                n += 1; total_gen += 1

        n_list.append(n); c_list.append(c)
        all_results.append({
            "task_id": task_id,
            "text": problem["text"],
            "test_list": problem.get("test_list", []),
            "completions": comps,
            "n": n, "c": c,
        })

        # 每10题保存一次
        if (i + 1) % 10 == 0:
            _save_results(out_file, model_name, config, all_results,
                         n_list, c_list, error_types, split,
                         len(problems), num_samples, temperature,
                         time.time() - start_time)

        # 进度日志
        if (i + 1) % 25 == 0 or (i + 1) == len(remaining):
            cur_p1 = total_passed / total_gen if total_gen else 0
            elapsed = time.time() - start_time
            eta = elapsed / (i + 1) * (len(remaining) - i - 1)
            avg_sec = elapsed / (i + 1)
            logger.info(
                f"  [{i+1}/{len(remaining)}] pass@1={cur_p1:.3f} | "
                f"avg={avg_sec:.1f}s/题 | "
                f"已用{elapsed/60:.1f}min | 剩余约{eta/60:.1f}min"
            )

    elapsed = time.time() - start_time

    # --- 最终保存 ---
    metrics = _save_results(out_file, model_name, config, all_results,
                            n_list, c_list, error_types, split,
                            len(problems), num_samples, temperature, elapsed)

    sep = "=" * 62
    logger.info(f"\n{sep}")
    logger.info(f"  模型:    {config['display_name']}")
    logger.info(f"  数据集:  MBPP {split} ({len(problems)} 题)")
    for k_str in ["pass@1", "pass@3"]:
        if k_str in metrics:
            logger.info(f"  {k_str}:  {metrics[k_str]:.4f}")
    logger.info(f"  耗时:    {elapsed/60:.1f} 分钟")
    logger.info(f"  错误:    {error_types}")
    logger.info(f"  输出:    {out_file}")
    logger.info(sep)
    return metrics


def _save_results(out_file, model_name, config, all_results,
                  n_list, c_list, error_types, split,
                  total_problems, num_samples, temperature, elapsed):
    """保存结果并返回metrics"""
    pass_k = {}
    for k in [1, 3]:
        vals = [compute_pass_at_k(n, c, k)
                for n, c in zip(n_list, c_list) if n >= k]
        if vals:
            pass_k[f"pass@{k}"] = float(np.mean(vals))

    metrics = {
        "model": model_name,
        "display_name": config["display_name"],
        "dataset": "MBPP",
        "split": split,
        "total_problems": total_problems,
        "completed_problems": len(all_results),
        "num_samples": num_samples,
        "temperature": temperature,
        **pass_k,
        "passed_total": sum(c_list),
        "elapsed_seconds": round(elapsed, 1),
        "error_distribution": error_types,
        "timestamp": datetime.now().isoformat(),
    }

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({"metrics": metrics, "results": all_results},
                  f, indent=2, ensure_ascii=False)
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
    parser.add_argument("--split", default="test")
    parser.add_argument("--max_problems", type=int, default=None)
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--no_resume", action="store_true",
                        help="不使用断点续跑，从头开始")
    args = parser.parse_args()

    m = run_mbpp(
        model_name=args.model,
        num_samples=args.num_samples,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        use_4bit=args.use_4bit,
        output_dir=args.output_dir,
        split=args.split,
        max_problems=args.max_problems,
        cache_dir=args.cache_dir,
        resume=not args.no_resume,
    )
    print(f"\n MBPP pass@1 = {m.get('pass@1', 'N/A'):.4f}")
