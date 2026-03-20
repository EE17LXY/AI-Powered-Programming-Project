"""
任务2: HumanEvalFix 缺陷检测与修复评测
数据集: bigcode/humanevalpack (Python)
任务: 给出有缺陷的Python代码，模型判断是否有缺陷 + 输出修复后的代码
指标: 检测准确率, 修复通过率 (pass@k), 修复质量

  python tasks/task2_defect_detection/run_humanevalfix.py \
      --model codellama \
      --use_4bit \
      --output_dir results/task2
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

import torch
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from models.model_loader import load_model, build_prompt, generate, extract_code_from_response

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# =================== 提示词 ===================
HUMANEVALFIX_SYSTEM = """You are an expert Python debugger. 
Given a Python function that may contain bugs, you will:
1. First determine if the code is buggy
2. If buggy, provide the corrected version

Always respond in the following exact format:
BUGGY: YES/NO
FIXED CODE:
```python
[your fixed code here, or original code if not buggy]
```"""


def format_humanevalfix_prompt(
    problem: dict,
    model_name: str,
    mode: str = "both"  # "detect", "fix", "both"
) -> str:
    """
    HumanEvalFix提示词

    HumanEvalPack数据集结构:
    - task_id: 题目ID
    - prompt: 函数签名和docstring
    - buggy_solution: 有缺陷的代码（函数体）
    - canonical_solution: 正确代码
    - bug_type: 缺陷类型
    - test: 测试代码
    - entry_point: 函数名
    """
    prompt_text = problem.get("prompt", "")
    buggy_code = problem.get("buggy_solution", problem.get("solution", ""))

    # 构建完整的有bug代码
    full_buggy = prompt_text + buggy_code

    if mode == "detect":
        instruction = f"""Examine this Python function and determine if it contains any bugs:

```python
{full_buggy}
```

Is this code buggy? Answer ONLY with "YES" or "NO"."""

    elif mode == "fix":
        instruction = f"""The following Python function contains a bug. Fix it:

```python
{full_buggy}
```

Provide only the corrected complete function implementation."""

    else:  # both
        instruction = f"""Analyze and fix the following Python function if it contains any bugs:

```python
{full_buggy}
```

First determine if it's buggy, then provide the fixed version.

Respond in this exact format:
BUGGY: [YES/NO]
FIXED CODE:
```python
[corrected function here]
```"""

    return build_prompt(model_name, instruction, system_prompt=HUMANEVALFIX_SYSTEM)


def parse_humanevalfix_response(response: str, problem: dict) -> dict:
    """
    解析模型响应，提取：
    1. 是否认为有bug? (0/1)
    2. 修复后的代码
    """
    result = {
        "is_buggy_pred": 1,  # 默认认为有bug
        "fixed_code": "",
    }

    buggy_match = re.search(r"BUGGY\s*:\s*(YES|NO)", response, re.IGNORECASE)
    if buggy_match:
        result["is_buggy_pred"] = 1 if "YES" in buggy_match.group(1).upper() else 0

    code = extract_code_from_response(response, language="python")

    if not code:
        lines = response.split('\n')
        code_lines = []
        in_code = False
        for line in lines:
            if line.strip().startswith("def ") or in_code:
                in_code = True
                code_lines.append(line)
        code = "\n".join(code_lines).strip()

    # 如果没有提取到代码，使用原始buggy代码
    if not code:
        code = problem.get("prompt", "") + problem.get("buggy_solution", "")

    result["fixed_code"] = code
    return result


def evaluate_fixed_code(problem: dict, fixed_code: str) -> dict:
    """评估修复后代码是否正确"""
    if not fixed_code.startswith("def ") and "def " in problem.get("prompt", ""):
        if not re.search(r"def\s+", fixed_code):
            fixed_code = problem["prompt"] + fixed_code

    full_code = fixed_code + "\n\n" + problem.get("test", "") + \
                f"\n\ncheck({problem.get('entry_point', 'solution')})"

    result = {"passed": False, "error": None}

    import multiprocessing as mp
    def _worker(q):
        try:
            exec(full_code, {})
            q.put({"passed": True, "error": None})
        except Exception as e:
            q.put({"passed": False, "error": f"{type(e).__name__}: {e}"})
    ctx = mp.get_context("fork")
    q = ctx.Queue()
    p = ctx.Process(target=_worker, args=(q,), daemon=True)
    p.start(); p.join(timeout=8)
    if p.is_alive():
        p.terminate(); p.join(timeout=2)
        if p.is_alive(): p.kill(); p.join()
        return {"passed": False, "error": "Timeout (>8s)"}
    return q.get() if not q.empty() else {"passed": False, "error": "No result"}


def run_humanevalfix(
    model_name: str,
    max_new_tokens: int = 512,
    temperature: float = 0.2,
    use_4bit: bool = False,
    output_dir: str = "results/task2",
    max_problems: int = None,
    cache_dir: str = None,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载模型
    logger.info(f"加载模型: {model_name}")
    model, tokenizer, config = load_model(
        model_name, use_4bit=use_4bit, cache_dir=cache_dir
    )

    # 加载数据集
    logger.info("加载 HumanEvalPack (HumanEvalFix) 数据集...")
    try:
        dataset = load_dataset("bigcode/humanevalpack", "python", split="test")
    except Exception:
        dataset = load_dataset("nuprl/HumanEval-solutions", split="test")

    problems = list(dataset)
    if max_problems:
        problems = problems[:max_problems]

    # HumanEvalFix: 所有样本的buggy_solution都是有bug的
    # 真实标签: 1
    logger.info(f"共 {len(problems)} 道题目")

    all_results = []
    y_true_detect = []   # 检测任务真实标签
    y_pred_detect = []   # 检测任务预测标签
    fix_pass_list = []   # 修复通过与否
    bug_type_stats = {}
    start_time = time.time()

    for i, problem in enumerate(tqdm(problems, desc=f"HumanEvalFix [{config['display_name']}]")):
        task_id = problem.get("task_id", str(i))
        bug_type = problem.get("bug_type", "unknown")

        try:
            prompt = format_humanevalfix_prompt(problem, model_name, mode="both")

            outputs = generate(
                model, tokenizer, prompt, config,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=(temperature > 0),
            )
            response = outputs[0]

            parsed = parse_humanevalfix_response(response, problem)

            # 评估修复
            fix_result = evaluate_fixed_code(problem, parsed["fixed_code"])

            # 记录
            y_true_detect.append(1)
            y_pred_detect.append(parsed["is_buggy_pred"])
            fix_pass_list.append(1 if fix_result["passed"] else 0)

            if bug_type not in bug_type_stats:
                bug_type_stats[bug_type] = {"total": 0, "detected": 0, "fixed": 0}
            bug_type_stats[bug_type]["total"] += 1
            if parsed["is_buggy_pred"] == 1:
                bug_type_stats[bug_type]["detected"] += 1
            if fix_result["passed"]:
                bug_type_stats[bug_type]["fixed"] += 1

            all_results.append({
                "task_id": task_id,
                "bug_type": bug_type,
                "true_label": 1,
                "predicted_buggy": parsed["is_buggy_pred"],
                "fixed_code": parsed["fixed_code"][:500],
                "fix_passed": fix_result["passed"],
                "fix_error": fix_result["error"],
                "response_preview": response[:200],
            })

        except Exception as e:
            logger.error(f"题目 {task_id} 出错: {e}")
            y_true_detect.append(1)
            y_pred_detect.append(0)
            fix_pass_list.append(0)

        if (i + 1) % 20 == 0:
            detect_acc = accuracy_score(y_true_detect, y_pred_detect)
            fix_rate = sum(fix_pass_list) / len(fix_pass_list)
            logger.info(f"  进度: {i+1}/{len(problems)}, "
                        f"检测Acc: {detect_acc:.3f}, 修复率: {fix_rate:.3f}")

    elapsed = time.time() - start_time

    # 指标
    detect_acc = accuracy_score(y_true_detect, y_pred_detect)
    detect_recall = sum(1 for t, p in zip(y_true_detect, y_pred_detect)
                        if t == 1 and p == 1) / sum(y_true_detect)
    fix_rate = sum(fix_pass_list) / len(fix_pass_list)

    metrics = {
        "model": model_name,
        "display_name": config["display_name"],
        "dataset": "HumanEvalFix",
        "total_problems": len(problems),
        "detection_accuracy": detect_acc,
        "detection_recall": detect_recall,
        "fix_pass_rate": fix_rate,
        "fix_pass_count": sum(fix_pass_list),
        "elapsed_seconds": elapsed,
        "bug_type_breakdown": bug_type_stats,
        "timestamp": datetime.now().isoformat(),
    }

    output_file = output_dir / f"humanevalfix_{model_name}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({"metrics": metrics, "results": all_results}, f, indent=2, ensure_ascii=False)

    logger.info(f"\n{'='*60}")
    logger.info(f"模型: {config['display_name']}")
    logger.info(f"检测准确率: {detect_acc:.4f}")
    logger.info(f"检测召回率 (bug recall): {detect_recall:.4f}")
    logger.info(f"修复通过率: {fix_rate:.4f} ({sum(fix_pass_list)}/{len(fix_pass_list)})")
    logger.info(f"Bug类型分布: {json.dumps(bug_type_stats, indent=2)}")
    logger.info(f"总耗时: {elapsed:.1f}秒")
    logger.info(f"结果已保存: {output_file}")
    logger.info(f"{'='*60}")

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HumanEvalFix缺陷检测与修复评测")
    parser.add_argument("--model", choices=["codellama", "qwen2coder"], required=True)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--use_4bit", action="store_true")
    parser.add_argument("--output_dir", default="/home/xyl/code_llm_survey/results/task2")
    parser.add_argument("--max_problems", type=int, default=None)
    parser.add_argument("--cache_dir", default=None)

    args = parser.parse_args()

    metrics = run_humanevalfix(
        model_name=args.model,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        use_4bit=args.use_4bit,
        output_dir=args.output_dir,
        max_problems=args.max_problems,
        cache_dir=args.cache_dir,
    )

    print(f"\n最终结果: 修复率={metrics['fix_pass_rate']:.4f}, 检测率={metrics['detection_accuracy']:.4f}")
