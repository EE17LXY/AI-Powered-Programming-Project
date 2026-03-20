"""
任务3: 代码可读性评估
数据集: se2p/code-readability-krod (Java代码) + 使用任务1生成的Python代码

评估方式：
1. 基于LLM打分: 让模型对代码可读性打分(1-5)
2. 静态指标: 圈复杂度、注释率、代码行数、命名规范等
3. 相关性分析: LLM评分 vs 静态指标 vs 人工标注

指标: Pearson/Spearman相关系数, MSE, MAE

  python tasks/task3_readability/run_readability.py \
      --model codellama \
      --use_4bit \
      --output_dir results/task3 \
      --task1_results results/task1/humaneval_codellama.json
"""

import os
import sys
import json
import argparse
import logging
import time
import re
import ast
from pathlib import Path
from datetime import datetime

import torch
import numpy as np
from datasets import load_dataset
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from models.model_loader import load_model, build_prompt, generate

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# =================== 提示词 ===================
READABILITY_SYSTEM = """You are an expert software engineer with extensive experience in code review.
Your task is to evaluate the readability of the given code snippet.

Readability Score Guidelines (1-5):
5 - Excellent: Clear naming, consistent style, well-commented, easy to understand at first glance
4 - Good: Mostly readable with minor improvements possible
3 - Average: Understandable but requires some effort, mixed quality
2 - Poor: Hard to read, confusing names or structure, lacks comments
1 - Very Poor: Extremely difficult to understand, very bad practices

Consider: variable/function naming, code structure, comments, complexity, consistency."""


def format_readability_prompt(code: str, language: str, model_name: str) -> str:
    if len(code) > 2048:
        code = code[:2048] + "\n... [truncated]"

    instruction = f"""Evaluate the readability of the following {language} code:

```{language.lower()}
{code}
```

Rate the readability on a scale of 1-5 (where 5 is most readable).

Respond in this EXACT format:
SCORE: [1-5]
REASONING: [brief 1-2 sentence explanation]"""

    return build_prompt(model_name, instruction, system_prompt=READABILITY_SYSTEM)


def parse_readability_score(response: str) -> dict:
    result = {"score": None, "reasoning": ""}

    # 提取SCORE
    score_match = re.search(r"SCORE\s*:\s*([1-5](?:\.[0-9])?)", response, re.IGNORECASE)
    if score_match:
        result["score"] = float(score_match.group(1))
    else:
        # 从响应中提取数字（？
        numbers = re.findall(r"\b([1-5])\b", response[:100])
        if numbers:
            result["score"] = float(numbers[0])

    # 提取REASONING
    reasoning_match = re.search(r"REASONING\s*:\s*(.+?)(?:\n|$)", response, re.IGNORECASE | re.DOTALL)
    if reasoning_match:
        result["reasoning"] = reasoning_match.group(1).strip()[:200]

    return result


# =================== 静态代码指标 ===================
def compute_static_metrics_python(code: str) -> dict:
    """计算Python代码静态指标"""
    metrics = {
        "loc": 0,          # 代码行数
        "cloc": 0,         # 注释行数
        "blank_lines": 0,  # 空白行数
        "comment_ratio": 0.0,  # 注释率
        "avg_line_length": 0.0,  # 平均行长度
        "max_line_length": 0,   # 最长行
        "num_functions": 0,     # 函数数量
        "avg_func_length": 0.0, # 平均函数长度
        "has_docstring": False, # 是否有docstring
        "naming_score": 0.0,   # 命名规范评分(0-1)
        "cyclomatic_complexity": 1,  # 圈复杂度
    }

    lines = code.split('\n')
    metrics["loc"] = len(lines)

    code_lines, comment_lines, blank_lines = 0, 0, 0
    line_lengths = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            blank_lines += 1
        elif stripped.startswith('#'):
            comment_lines += 1
        else:
            code_lines += 1
        line_lengths.append(len(line))

    metrics["cloc"] = comment_lines
    metrics["blank_lines"] = blank_lines
    metrics["comment_ratio"] = comment_lines / max(metrics["loc"], 1)
    metrics["avg_line_length"] = np.mean(line_lengths) if line_lengths else 0
    metrics["max_line_length"] = max(line_lengths) if line_lengths else 0

    # 计算圈复杂度
    complexity_keywords = [
        'if ', 'elif ', 'else:', 'for ', 'while ',
        'except', 'and ', 'or ', 'not ', 'with '
    ]
    cc = 1
    for line in lines:
        for kw in complexity_keywords:
            if kw in line:
                cc += 1
    metrics["cyclomatic_complexity"] = cc

    try:
        tree = ast.parse(code)

        functions = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
        metrics["num_functions"] = len(functions)

        for func in functions:
            if (func.body and isinstance(func.body[0], ast.Expr) and
                    isinstance(func.body[0].value, ast.Constant) and
                    isinstance(func.body[0].value.value, str)):
                metrics["has_docstring"] = True
                break

        # 命名规范检查
        names = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                names.append(node.name)
            elif isinstance(node, ast.Name):
                names.append(node.id)

        if names:
            # snake_case: 全小写+下划线
            snake_case_count = sum(1 for n in names
                                   if re.match(r'^[a-z][a-z0-9_]*$', n) or len(n) == 1)
            # 排除Python关键字和内置名
            python_builtins = {'print', 'len', 'range', 'int', 'str', 'list', 'dict',
                               'None', 'True', 'False', 'self', 'cls', 'return', 'type'}
            valid_names = [n for n in names if n not in python_builtins]
            if valid_names:
                metrics["naming_score"] = snake_case_count / len(valid_names)

    except SyntaxError:
        pass

    return metrics


def compute_static_metrics_java(code: str) -> dict:
    """计算Java代码静态指标"""
    metrics = {
        "loc": 0,
        "cloc": 0,
        "blank_lines": 0,
        "comment_ratio": 0.0,
        "avg_line_length": 0.0,
        "max_line_length": 0,
        "num_methods": 0,
        "has_javadoc": False,
        "cyclomatic_complexity": 1,
        "naming_score": 0.0,
    }

    lines = code.split('\n')
    metrics["loc"] = len(lines)

    comment_lines, blank_lines = 0, 0
    in_block_comment = False
    line_lengths = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            blank_lines += 1
        elif "/*" in stripped or in_block_comment:
            comment_lines += 1
            if "*/" in stripped:
                in_block_comment = False
            elif "/*" in stripped and "*/" not in stripped:
                in_block_comment = True
        elif stripped.startswith("//"):
            comment_lines += 1
        line_lengths.append(len(line))

    metrics["cloc"] = comment_lines
    metrics["blank_lines"] = blank_lines
    metrics["comment_ratio"] = comment_lines / max(len(lines), 1)
    metrics["avg_line_length"] = np.mean(line_lengths) if line_lengths else 0
    metrics["max_line_length"] = max(line_lengths) if line_lengths else 0

    # 方法数量
    method_pattern = re.compile(
        r'(?:public|private|protected|static|\s)+[\w<>\[\]]+\s+(\w+)\s*\([^)]*\)\s*(?:throws\s+\w+\s*)?\{'
    )
    methods = method_pattern.findall(code)
    metrics["num_methods"] = len(methods)

    # JavaDoc
    metrics["has_javadoc"] = "/**" in code

    # 圈复杂度
    cc_keywords = ['if', 'else if', 'for', 'while', 'case', 'catch', '&&', '||', '?', 'switch']
    cc = 1
    for line in lines:
        for kw in cc_keywords:
            if kw in line:
                cc += 1
    metrics["cyclomatic_complexity"] = min(cc, 50)

    # 命名规范
    camel_names = method_pattern.findall(code)
    if camel_names:
        camel_count = sum(1 for n in camel_names
                         if re.match(r'^[a-z][a-zA-Z0-9]*$', n))
        metrics["naming_score"] = camel_count / len(camel_names)

    return metrics


def readability_score_from_static_python(metrics: dict) -> float:
    """
    根据静态指标计算综合可读性评分(1-5)
    """
    score = 3.0

    # 注释率 (0-0.3比较不错)
    cr = metrics.get("comment_ratio", 0)
    if cr > 0.15:
        score += 0.5
    elif cr < 0.05:
        score -= 0.5

    # 命名规范 (越高越好)
    ns = metrics.get("naming_score", 0.5)
    score += (ns - 0.5) * 1.0

    # 圈复杂度 (越低越好)
    cc = metrics.get("cyclomatic_complexity", 1)
    if cc <= 5:
        score += 0.5
    elif cc >= 15:
        score -= 1.0
    elif cc >= 10:
        score -= 0.5

    # 有docstring加分
    if metrics.get("has_docstring", False):
        score += 0.3

    # 最大行长度 (PEP8建议<=79)
    max_len = metrics.get("max_line_length", 0)
    if max_len > 120:
        score -= 0.3
    elif max_len <= 79:
        score += 0.2

    # 平均行长度 (25-60比较合适)
    avg_len = metrics.get("avg_line_length", 40)
    if 25 <= avg_len <= 60:
        score += 0.2

    return max(1.0, min(5.0, score))


def run_readability(
    model_name: str,
    max_new_tokens: int = 64,
    use_4bit: bool = False,
    output_dir: str = "results/task3",
    task1_results_file: str = None,  # 使用任务1生成的代码
    max_java_samples: int = 200,     # Java数据集
    cache_dir: str = None,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载模型
    logger.info(f"加载模型: {model_name}")
    model, tokenizer, config = load_model(
        model_name, use_4bit=use_4bit, cache_dir=cache_dir
    )

    all_evaluations = []

    # ==========  Java数据集 (se2p/code-readability-krod) ==========
    logger.info("\n--- Java数据集 ---")
    try:
        java_dataset = load_dataset("se2p/code-readability-krod", split="train")
        java_samples = list(java_dataset)

        # 采样
        if len(java_samples) > max_java_samples:
            import random
            random.seed(42)
            java_samples = random.sample(java_samples, max_java_samples)

        logger.info(f"Java样本数: {len(java_samples)}")

        java_llm_scores = []
        java_human_scores = []
        java_static_scores = []

        for i, sample in enumerate(tqdm(java_samples, desc=f"Java可读性 [{config['display_name']}]")):
            code = sample.get("code_snippet", sample.get("code", ""))
            human_score = sample.get("readability_score", sample.get("score", 3.0))

            # 归一化人工评分到1-5
            if human_score > 5:
                human_score = human_score / (max(s.get("readability_score", 5)
                                                 for s in java_samples[:10]) / 5.0)
                human_score = max(1.0, min(5.0, human_score))

            # LLM评分
            try:
                prompt = format_readability_prompt(code, "Java", model_name)
                outputs = generate(
                    model, tokenizer, prompt, config,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                )
                parsed = parse_readability_score(outputs[0])
                llm_score = parsed["score"]
            except Exception as e:
                logger.debug(f"样本 {i} 出错: {e}")
                llm_score = None

            # 静态指标评分
            static_metrics = compute_static_metrics_java(code)
            static_score = readability_score_from_static_python(static_metrics)

            if llm_score is not None:
                java_llm_scores.append(llm_score)
                java_human_scores.append(human_score)
                java_static_scores.append(static_score)

                all_evaluations.append({
                    "source": "java_dataset",
                    "code_preview": code[:300],
                    "human_score": human_score,
                    "llm_score": llm_score,
                    "static_score": static_score,
                    "static_metrics": static_metrics,
                    "llm_reasoning": parsed.get("reasoning", ""),
                })

        # Java部分相关性
        if len(java_llm_scores) >= 10:
            pearson_llm_human, _ = pearsonr(java_llm_scores, java_human_scores)
            spearman_llm_human, _ = spearmanr(java_llm_scores, java_human_scores)
            pearson_static_human, _ = pearsonr(java_static_scores, java_human_scores)
            spearman_static_human, _ = spearmanr(java_static_scores, java_human_scores)

            logger.info(f"\nJava数据集相关性:")
            logger.info(f"  LLM评分 vs 人工评分: Pearson={pearson_llm_human:.4f}, "
                        f"Spearman={spearman_llm_human:.4f}")
            logger.info(f"  静态评分 vs 人工评分: Pearson={pearson_static_human:.4f}, "
                        f"Spearman={spearman_static_human:.4f}")

    except Exception as e:
        logger.error(f"加载Java数据集失败: {e}")
        java_llm_scores, java_human_scores = [], []

    # ========== 任务1生成的Python代码 ==========
    python_evaluations = []
    if task1_results_file and os.path.exists(task1_results_file):
        logger.info(f"\n--- 任务1生成的Python代码 ({task1_results_file}) ---")

        with open(task1_results_file, 'r') as f:
            task1_data = json.load(f)

        task1_results = task1_data.get("results", [])
        logger.info(f"共 {len(task1_results)} 个任务1结果")

        for item in tqdm(task1_results[:100], desc="Python代码可读性"):
            completions = item.get("completions", [])
            if not completions:
                continue

            # 取第一个通过的completion，或第一个completion
            best_completion = None
            for comp in completions:
                if comp.get("passed", False):
                    best_completion = comp
                    break
            if best_completion is None and completions:
                best_completion = completions[0]

            if best_completion is None:
                continue

            code = best_completion.get("completion", best_completion.get("raw_output", ""))
            if not code or len(code) < 20:
                continue

            # LLM可读性评分
            try:
                full_code = item.get("problem", "") + code
                prompt = format_readability_prompt(full_code, "Python", model_name)
                outputs = generate(
                    model, tokenizer, prompt, config,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                )
                parsed = parse_readability_score(outputs[0])
                llm_score = parsed["score"]
            except Exception as e:
                logger.debug(f"可读性评分出错: {e}")
                llm_score = None

            if llm_score is None:
                continue

            # 静态指标
            static_metrics = compute_static_metrics_python(code)
            static_score = readability_score_from_static_python(static_metrics)
            passed = best_completion.get("passed", False)

            python_evaluations.append({
                "task_id": item.get("task_id", ""),
                "code": code[:500],
                "passed": passed,
                "llm_score": llm_score,
                "static_score": static_score,
                "static_metrics": static_metrics,
                "llm_reasoning": parsed.get("reasoning", ""),
            })

        all_evaluations.extend([{**e, "source": "python_generated"} for e in python_evaluations])
        logger.info(f"Python代码评估完成: {len(python_evaluations)} 个")

        if python_evaluations:
            passed_scores = [e["llm_score"] for e in python_evaluations if e["passed"] and e["llm_score"]]
            failed_scores = [e["llm_score"] for e in python_evaluations if not e["passed"] and e["llm_score"]]

            if passed_scores and failed_scores:
                logger.info(f"\nPython代码可读性对比:")
                logger.info(f"  通过的代码: 平均LLM评分={np.mean(passed_scores):.2f}")
                logger.info(f"  未通过的代码: 平均LLM评分={np.mean(failed_scores):.2f}")

    # 统计
    elapsed = time.time()

    all_llm_scores = [e["llm_score"] for e in all_evaluations if "llm_score" in e and e["llm_score"]]
    all_static_scores = [e["static_score"] for e in all_evaluations if "static_score" in e]

    metrics = {
        "model": model_name,
        "display_name": config["display_name"],
        "total_evaluated": len(all_evaluations),
        "avg_llm_score": float(np.mean(all_llm_scores)) if all_llm_scores else 0,
        "std_llm_score": float(np.std(all_llm_scores)) if all_llm_scores else 0,
        "avg_static_score": float(np.mean(all_static_scores)) if all_static_scores else 0,
        "timestamp": datetime.now().isoformat(),
    }

    if java_llm_scores and java_human_scores and len(java_llm_scores) >= 10:
        pearson_r, pearson_p = pearsonr(java_llm_scores, java_human_scores)
        spearman_r, spearman_p = spearmanr(java_llm_scores, java_human_scores)
        mae = float(np.mean(np.abs(np.array(java_llm_scores) - np.array(java_human_scores))))
        mse = float(np.mean((np.array(java_llm_scores) - np.array(java_human_scores))**2))

        metrics.update({
            "java_samples": len(java_llm_scores),
            "pearson_llm_human": float(pearson_r),
            "pearson_p_value": float(pearson_p),
            "spearman_llm_human": float(spearman_r),
            "spearman_p_value": float(spearman_p),
            "mae_llm_human": mae,
            "mse_llm_human": mse,
        })

    if python_evaluations:
        py_passed = [e["llm_score"] for e in python_evaluations if e["passed"] and e["llm_score"]]
        py_failed = [e["llm_score"] for e in python_evaluations if not e["passed"] and e["llm_score"]]
        metrics["python_passed_avg_readability"] = float(np.mean(py_passed)) if py_passed else 0
        metrics["python_failed_avg_readability"] = float(np.mean(py_failed)) if py_failed else 0
        metrics["python_readability_correctness_correlation"] = \
            "readable code tends to be correct" if metrics.get("python_passed_avg_readability", 0) > \
            metrics.get("python_failed_avg_readability", 0) else "no clear correlation"

    # 保存结果
    output_file = output_dir / f"readability_{model_name}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({
            "metrics": metrics,
            "java_evaluations": [e for e in all_evaluations if e.get("source") == "java_dataset"],
            "python_evaluations": python_evaluations[:100],
        }, f, indent=2, ensure_ascii=False, default=str)

    logger.info(f"\n{'='*60}")
    logger.info(f"模型: {config['display_name']}")
    logger.info(f"总评估样本: {len(all_evaluations)}")
    logger.info(f"平均LLM可读性评分: {metrics.get('avg_llm_score', 0):.2f}/5.0")
    if "pearson_llm_human" in metrics:
        logger.info(f"Pearson相关 (LLM vs 人工): {metrics['pearson_llm_human']:.4f}")
        logger.info(f"Spearman相关 (LLM vs 人工): {metrics['spearman_llm_human']:.4f}")
        logger.info(f"MAE: {metrics['mae_llm_human']:.4f}")
    logger.info(f"结果已保存: {output_file}")
    logger.info(f"{'='*60}")

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="代码可读性评估")
    parser.add_argument("--model", choices=["codellama", "qwen2coder"], required=True)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--use_4bit", action="store_true")
    parser.add_argument("--output_dir", default="/home/xyl/code_llm_survey/results/task3")
    parser.add_argument("--task1_results", default=None,
                        help="任务1的结果JSON文件路径，用于评估生成代码的可读性")
    parser.add_argument("--max_java_samples", type=int, default=200)
    parser.add_argument("--cache_dir", default=None)

    args = parser.parse_args()

    metrics = run_readability(
        model_name=args.model,
        max_new_tokens=args.max_new_tokens,
        use_4bit=args.use_4bit,
        output_dir=args.output_dir,
        task1_results_file=args.task1_results,
        max_java_samples=args.max_java_samples,
        cache_dir=args.cache_dir,
    )

    print(f"\n最终结果: 平均LLM可读性={metrics.get('avg_llm_score', 0):.2f}/5.0")
    if "pearson_llm_human" in metrics:
        print(f"  Pearson相关: {metrics['pearson_llm_human']:.4f}")
