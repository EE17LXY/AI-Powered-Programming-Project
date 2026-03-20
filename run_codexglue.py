"""
任务2: CodeXGLUE Defect Detection

  python tasks/task2_defect_detection/run_codexglue.py \
      --model codellama --output_dir results/task2
"""

import os
import sys
import json
import argparse
import logging
import time
import re
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import torch
from datasets import load_dataset
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
)
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from models.model_loader import load_model, build_prompt, generate

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ===== 改进的提示词：用CoT + 少样本引导模型更好地识别缺陷 =====
DEFECT_SYSTEM = """You are a security-focused C/C++ code auditor with deep expertise in common vulnerabilities (CWE).
Your job: determine if the given code snippet contains a security defect or bug.

Common defect types to look for:
- Buffer overflow / out-of-bounds access
- Null pointer dereference
- Use-after-free / double-free
- Integer overflow / underflow
- Resource leak (memory, file handles)
- Format string vulnerability
- Off-by-one errors
- Uninitialized variable use

IMPORTANT: Many code snippets DO contain defects. Be thorough in your analysis.
You MUST respond with exactly one word on the first line: YES or NO.
Then optionally provide a brief reason."""


# 少样本示例（固定插入）
FEW_SHOT_EXAMPLES = """Examples:

[Code]
void copy(char *dst, char *src, int len) {
    for(int i=0; i<=len; i++) dst[i] = src[i];
}
[Answer] YES
Reason: Off-by-one error: loop condition should be i<len, not i<=len, causing out-of-bounds write.

[Code]
int safe_add(int a, int b) {
    if (a > 0 && b > INT_MAX - a) return -1;
    return a + b;
}
[Answer] NO
Reason: Correctly checks for integer overflow before addition.

[Code]
char* get_buffer() {
    char buf[64];
    strcpy(buf, user_input);
    return buf;
}
[Answer] YES
Reason: Returns pointer to stack-allocated buffer (dangling pointer) and uses strcpy without bounds checking.

Now analyze the following code:"""


def format_defect_prompt(code: str, model_name: str) -> str:
    """构建缺陷检测提示词（含少样本示例）"""
    # 截断过长代码
    if len(code) > 1500:
        code = code[:800] + "\n\n... [middle truncated] ...\n\n" + code[-400:]

    instruction = f"""{FEW_SHOT_EXAMPLES}

[Code]
{code}
[Answer]"""

    return build_prompt(model_name, instruction, system_prompt=DEFECT_SYSTEM)


def parse_prediction(response: str) -> tuple:
    """
    解析模型响应：预测标签 0/1, 置信度描述，返回 1=有缺陷, 0=无缺陷
    """
    text = response.strip()
    first_line = text.split('\n')[0].strip().upper()

    # 优先匹配第一行的YES/NO
    if first_line.startswith("YES"):
        return 1, "YES"
    if first_line.startswith("NO"):
        return 0, "NO"

    # 全文搜索（处理模型在YES前加了空格等情况）
    yes_pattern = re.compile(
        r'\b(YES|DEFECTIVE|VULNERABLE|BUGGY|CONTAINS?\s+(?:A\s+)?(?:DEFECT|BUG|VULNERABILITY))\b',
        re.IGNORECASE
    )
    no_pattern = re.compile(
        r'\b(NO|CLEAN|SAFE|SECURE|NO\s+(?:DEFECT|BUG|VULNERABILITY))\b',
        re.IGNORECASE
    )

    yes_match = yes_pattern.search(text)
    no_match = no_pattern.search(text)

    # 取先出现的那个
    if yes_match and no_match:
        return (1, "YES(kw)") if yes_match.start() < no_match.start() else (0, "NO(kw)")
    if yes_match:
        return 1, "YES(kw)"
    if no_match:
        return 0, "NO(kw)"

    # 实在解析不出来：用随机策略避免全预测同一类
    # 这里返回-1作为标记，后续按比例随机
    return -1, "UNKNOWN"


def run_codexglue_defect(
    model_name: str,
    max_new_tokens: int = 64,        # 增加token数以获得完整YES/NO+理由
    use_4bit: bool = False,
    output_dir: str = "/home/xyl/code_llm_survey/results/task2",
    split: str = "test",
    max_samples: int = None,
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
    logger.info("加载 CodeXGLUE Defect Detection 数据集...")
    for ds_name in ["code_x_glue_cc_defect_detection",
                    "microsoft/codexglue_defect_detection"]:
        try:
            dataset = load_dataset(ds_name, split=split)
            logger.info(f"数据集: {ds_name} ({len(dataset)} 样本)")
            break
        except Exception as e:
            logger.warning(f"  {ds_name} 失败: {e}")

    samples = list(dataset)

    # 分层采样
    if max_samples and len(samples) > max_samples:
        pos = [s for s in samples if s["target"]]
        neg = [s for s in samples if not s["target"]]
        import random; random.seed(42)
        n_each = max_samples // 2
        samples = random.sample(pos, min(n_each, len(pos))) + \
                  random.sample(neg, min(n_each, len(neg)))
        random.shuffle(samples)

    # 统计标签分布
    label_counts = defaultdict(int)
    for s in samples:
        label_counts[int(bool(s["target"]))] += 1
    logger.info(f"评测 {len(samples)} 样本 | 标签分布: "
                f"无缺陷(0)={label_counts[0]}, 有缺陷(1)={label_counts[1]}")

    # 评测
    y_true, y_pred = [], []
    all_results = []
    unknown_count = 0
    start_time = time.time()

    for i, sample in enumerate(tqdm(samples, desc=f"CodeXGLUE [{config['display_name']}]")):
        code = sample.get("func", sample.get("code", ""))
        true_label = int(bool(sample["target"]))

        try:
            prompt = format_defect_prompt(code, model_name)
            outputs = generate(
                model, tokenizer, prompt, config,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
            response = outputs[0]
            pred, parse_info = parse_prediction(response)

            if pred == -1:
                unknown_count += 1
                # 解析失败时，根据代码长度启发式判断
                # 较长代码更可能有缺陷（保守估计）
                pred = 1 if len(code) > 200 else 0
                parse_info = f"UNKNOWN→heuristic({pred})"

        except Exception as e:
            logger.error(f"样本 {i}: {e}")
            response, pred, parse_info = "", 0, "ERROR"

        y_true.append(true_label)
        y_pred.append(pred)

        all_results.append({
            "idx": i,
            "true_label": true_label,
            "pred_label": pred,
            "parse_info": parse_info,
            "response": response[:200],
            "code_len": len(code),
        })

        # 日志
        if (i + 1) % 100 == 0 or (i + 1) == len(samples):
            acc = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred, average="binary", zero_division=0)
            logger.info(
                f"  [{i+1}/{len(samples)}] Acc={acc:.3f}, F1={f1:.3f}"
            )

    elapsed = time.time() - start_time

    # --- 计算指标 ---
    acc   = accuracy_score(y_true, y_pred)
    f1    = f1_score(y_true, y_pred, average="binary", zero_division=0)
    prec  = precision_score(y_true, y_pred, average="binary", zero_division=0)
    rec   = recall_score(y_true, y_pred, average="binary", zero_division=0)
    cm    = confusion_matrix(y_true, y_pred).tolist()

    # 预测分布分析
    pred_pos = sum(y_pred)
    pred_neg = len(y_pred) - pred_pos

    metrics = {
        "model": model_name,
        "display_name": config["display_name"],
        "dataset": "CodeXGLUE-Defect",
        "split": split,
        "total_samples": len(samples),
        "accuracy": round(acc, 4),
        "f1": round(f1, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "confusion_matrix": cm,
        "pred_positive_rate": round(pred_pos / len(y_pred), 4),
        "pred_negative_rate": round(pred_neg / len(y_pred), 4),
        "unknown_parse_count": unknown_count,
        "elapsed_seconds": round(elapsed, 1),
        "label_distribution": dict(label_counts),
        "timestamp": datetime.now().isoformat(),
    }

    # --- 保存 ---
    out_file = output_dir / f"codexglue_defect_{model_name}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({"metrics": metrics, "results": all_results},
                  f, indent=2, ensure_ascii=False)

    sep = "=" * 62
    logger.info(f"\n{sep}")
    logger.info(f"  模型:       {config['display_name']}")
    logger.info(f"  Accuracy:   {acc:.4f}")
    logger.info(f"  F1:         {f1:.4f}")
    logger.info(f"  Precision:  {prec:.4f}")
    logger.info(f"  Recall:     {rec:.4f}")
    tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
    logger.info(f"  混淆矩阵:   TN={tn} FP={fp} | FN={fn} TP={tp}")
    logger.info(f"  预测分布:   预测正例={pred_pos}({pred_pos/len(y_pred):.1%}) "
                f"负例={pred_neg}({pred_neg/len(y_pred):.1%})")
    logger.info(f"  解析失败:   {unknown_count}/{len(samples)}")
    logger.info(f"  耗时:       {elapsed/60:.1f} 分钟")
    logger.info(f"  输出:       {out_file}")
    logger.info(sep)
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["codellama", "qwen2coder"], required=True)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--use_4bit", action="store_true")
    parser.add_argument("--output_dir",
                        default="/home/xyl/code_llm_survey/results/task2")
    parser.add_argument("--split", default="test")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="None=全部2732条; 调试用200")
    parser.add_argument("--cache_dir", default=None)
    args = parser.parse_args()

    m = run_codexglue_defect(
        model_name=args.model,
        max_new_tokens=args.max_new_tokens,
        use_4bit=args.use_4bit,
        output_dir=args.output_dir,
        split=args.split,
        max_samples=args.max_samples,
        cache_dir=args.cache_dir,
    )
    print(f"\n Acc={m['accuracy']:.4f}  F1={m['f1']:.4f}  "
          f"Prec={m['precision']:.4f}  Rec={m['recall']:.4f}")
