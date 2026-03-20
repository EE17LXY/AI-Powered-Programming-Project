# 面向高质量软件开发的智能编程辅助技术研究与原型实现

## 项目简介
本项目针对当前AI编程助手爆发式涌现、模型性能难以客观衡量的问题，选取**CodeLlama-7B**与**Qwen2.5-Coder-7B**两款主流代码大模型，在统一实验环境下完成**代码补全、缺陷检测、代码可读性评估**三大核心任务的对比评测，为开发者选择智能编程工具提供量化依据。

## 研究内容
- 调研并复现面向代码生成与代码理解的大语言模型算法
- 搭建标准化评测框架，支持多数据集、多任务自动评估
- 完成代码生成、缺陷检测与修复、可读性打分对比实验
- 输出可复现的实验流程、数据结果与技术分析

## 评测模型
### 1. CodeLlama-7B-Instruct
### 2. Qwen2.5-Coder-7B-Instruct

## 评测任务与数据集
### 1. 代码补全
- HumanEval（164题 Python）
- MBPP（500题 Python）
- 指标：pass@1 / pass@3

### 2. 缺陷检测与修复
- CodeXGLUE（2732段 C/C++ 漏洞检测）
- HumanEvalFix（164个 Python Bug 修复）
- 指标：准确率、精确率、召回率、F1、修复通过率

### 3. 代码可读性评估
- 数据集：code-readability-krod（Java/Python）
- 指标：LLM打分（1-5分）、Pearson/Spearman相关系数

## 代码框架
- `model_loader.py`：模型加载、GPU显存管理、支持4bit量化
- `run_humaneval.py`：HumanEval代码生成评测
- `run_mbpp.py`：MBPP代码生成评测
- `run_codexglue.py`：C/C++缺陷检测
- `run_humanevalfix.py`：Python缺陷修复
- `run_readability.py`：代码可读性评估

## 实验结果
### 代码补全（pass@1）
- HumanEval：CodeLlama 23.17% | Qwen2.5 53.05%
- MBPP：CodeLlama 43.40% | Qwen2.5 76.80%

### 缺陷修复（HumanEvalFix）
- 修复通过率：CodeLlama 35.98% | Qwen2.5 85.98%

### 缺陷检测（CodeXGLUE）
- 准确率：CodeLlama 53.88% | Qwen2.5 54.32%
- F1：CodeLlama 0.147 | Qwen2.5 0.204

### 可读性评估
- Pearson相关系数：CodeLlama 0.110 | Qwen2.5 0.290
