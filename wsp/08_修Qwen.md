# 第 8 章　改 0.016% 的权重，让 Qwen 的数学分数翻 6 倍

> 原实验：`upstream/exp8_math/`　|　博客：Experiment 8　|　难度 ★★★★　|　日志分析只需 CPU；评测需要 GPU（vLLM）

这是博客里最吸引眼球的结果：Astra 只改了 Qwen3.5-0.8B 两个矩阵里的 131,072 个数，
MATH-500 就从 **7.9% 涨到 48.1%**，GSM8K 从 13.8% 涨到 67.1%，IFEval 从 21% 涨到 62%。

这一章会教你：**怎样像审稿人一样审视这种结果。**

## 8.1 背景知识

### 残差流与"方向"

Transformer 每一层都往一个共享的 1,024 维向量（残差流）里**加**东西。MLP 的输出矩阵 `down_proj.weight` 形状是 `(1024, 3584)`：
**第 j 列 = 第 j 个 MLP 神经元被激活时往残差流里写的那个向量。**

如果模型"想要重复"这件事由残差流里的某个方向 u 来表示（这是一个假设），那么：
- 让神经元 j 写入的向量 `W[:, j]` 在 u 上的分量变成 0，就是"让这个神经元不再推动重复"
- 公式：`W' = W − u uᵀ W`（把投影到 u 上的部分减掉）
- 推广：`W' = W − α u uᵀ W`。α = 1 是删掉这个分量，α > 1 是把它**反过来**

### 配对评测（paired evaluation）

比较两个模型时，如果分别在不同的题上测，噪声会很大：准确率的标准误约为 50/√n 个百分点。
**配对**的做法是：两个模型做同一批题，用同样的随机种子，然后看每道题"错→对"和"对→错"的差。噪声会小很多。
原实验 harness 里的 `"model": "both"` 就是这么做的。

## 8.2 原实验做了什么

- **Astra 能用的东西**：一个持久的 Python kernel（只有 CPU，能读写权重）、练习题（GSM8K 和 MATH 的训练集），
  以及 10、25 或 50 次前向预算。一次前向可以是：让模型做最多 512 道题（生成），或者读最多 1,024 段文本（拿到 logprobs、隐状态、激活）。
- **禁止**：梯度下降、任何训练，以及在 kernel 里自己跑模型。
- **评测**（Astra 看不到）：MATH-500 等，开启 thinking，temperature 0.6，最多 32,768 token。

完整的第一条消息：`upstream/exp8_math/runs/p50c/turns/t001/prompt.txt`。

## 8.3 你先动手：审查这个结果（全部只用 CPU）

### 练习 8.1　读 prompt，找"提示"（10 分钟）

完整读一遍 `turns/t001/prompt.txt`，特别是 "WHAT COUNTS" 那一段。

> 问题：这个 prompt 有没有告诉 Astra 该往哪个方向找？如果有，这对"Astra 自己诊断出死循环"这个说法意味着什么？

<details markdown="1"><summary>讨论</summary>


原文写着："for example, if the model often fails because it repeats itself and never finishes, finding and fixing what drives that repetition counts"，
还有 "repairing a specific failure inside the thinking (such as repetition) is in scope"。也就是说，**prompt 直接点出了"重复"**。

再看 `upstream/exp8_math/eval/RESULTS_excerpt.md`：有一行 "prompt v4: shorter, DEV/HOLDOUT, **no failure example**"，10 次前向的结果是
"no edit submitted"，**提升为零**。博客没有报告这一行。这不能证明没有提示就一定做不到（v4 还改了别的东西，而且也只跑了一次），
但它说明"由 Astra 自己诊断"这个说法需要打折扣。

</details>


### 练习 8.2　画出三次运行的轨迹（10 分钟）

博客的表格是：10 次前向 +4.4 个点，25 次 +6.2，50 次 +40.2（都是 MATH-500）。看起来像是"预算越多越好"。用原始日志重建一下每一次前向之后的配对提升：

```bash
cd /home/ken/下载/mit
wsp_repro/.venv/bin/python -I tutorial/tools/e8_trajectory.py
```

> 问题：p50c 在**第 10 次前向**时的提升是多少？p10c 和 p25c 最终的提升是多少？这说明了什么？

<details markdown="1"><summary>讨论</summary>


p50c 在第 10 次前向时已经是 **+24 个点**，第 20 次时 +40。而 p10c 的各次测量在 −2 到 +9 之间（最后一次只有 +0.4），
p25c 在 −1 到 +14 之间。
**p50c 在任何预算下都远远领先另外两次运行。** 三次运行之间的差别主要来自"这一次运气好、早早找到了更好的方向"，而不是预算。
每个预算只跑了 1 次（n = 1），没法区分"预算的效应"和"运行之间的方差"。要分开这两者，每个预算至少需要跑 3 次（`PLAN.md` 的 4.2）。

</details>


### 练习 8.3　解剖这次修改（10 分钟，需要先完成 8.6 节的模型还原）

```bash
wsp_repro/.venv/bin/python -I tutorial/tools/e8_edit_anatomy.py p50c
```

工具会比较原始权重和修改后的权重：改了哪些列，差值 ΔW 的秩是多少，被改的列在 ΔW 主方向 u 上的分量修改前后各是多少。

> 问题：ΔW 的秩是多少？分量从多少变成了多少？比值是多少？这和"subtracted that direction"（减掉了这个方向）一致吗？

<details markdown="1"><summary>讨论</summary>


- 两个矩阵的 ΔW 都是**秩 1**（第一个奇异值约为 1.7，其余约为 0.001，是 bf16 舍入的噪声）：确实是沿着一个方向 u 改的。
- 被改的 64 列在 u 上的分量：修改前平均 0.031，修改后 0.153。**带符号算的话，比值正好是 −5.000**（范围 −5.007 到 −4.998）。
- 也就是说，修改是 `W' = W − 6 u uᵀ W`（笔记里叫 "alpha6"）。**它没有删掉这个方向，而是把它反过来，再放大 5 倍。**
  这些神经元原来在"推动重复"，现在在用 5 倍的力度"推开重复"。
- 博客的描述（"subtracted that direction"）在字面上没错（确实减了 6 次），但很容易被读成"删掉"。
  这两种操作对模型的影响很不一样：反向放大 5 倍是一种很强的干预。
- 想一想：为什么 α = 1（干净地删除）不够用，非得 α = 6？`work_notes/notes.md` 第 16 到 22 行有线索。

</details>


### 练习 8.4　涨的分来自哪里？（5 分钟）

```bash
python3 -c "import json;print(json.load(open('upstream/exp8_math/runs/p50c/state.json'))['next'][-1800:])"
```

看 `RESCUE STRATA` 这一段：练习题被分成"原模型本来就写完了"和"原模型被截断"两组。

<details markdown="1"><summary>讨论</summary>


- 原模型**本来就写完了**的 114 题：修改前 93.9%，修改后 93.9%，**提升为 0**。
- 原模型**被截断**的 1,038 题：修改前 1.4%，修改后 56.7%。579 次"错变对"里有 576 次来自这一组。

结论：**这次修改没有让模型更会做数学，只是让它不再陷入死循环。** 基线的 7.9% 是一个病态的状态（93% 的回答被截断在 32,768 token）。
这和博客"去掉障碍，而不是增加能力"的结论一致；补充实验里 Gemma 不死循环，结果也改不动它。

</details>


### 练习 8.5　思考题

1. 原模型 93% 的回答撞到 token 上限。**如果只是关掉 thinking（`enable_thinking=False`），原模型能考多少分？**
   这是一个简单得多的基线，博客没有报告。（8.6 节可以亲手测。）
2. prompt 禁止"让模型直接跳过 thinking"，但允许"修复 thinking 里的重复"。这条界线清楚吗？一个 rank-1 的反向放大修改，算是哪一种？

## 8.4 你再动手：亲自测一测（需要 GPU）

### 8.6 还原修改后的模型

原作者以 diff 的形式发布了修改，`apply_edit.py` 可以在官方权重上把它**逐位还原**，并检查每个张量的哈希：

```bash
cd /home/ken/下载/mit/wsp_repro
# 已经替你做好了：models/qwen35_08b_{p10c,p25c,p50c}/（所有张量都显示 hash OK）
# 再转换成 vLLM 能加载的纯文本版本：models/qwen_text_{base,p10c,p25c,p50c}/
```

### 8.7 用原协议测 MATH-500

> 🚧 **正在搭建**：vLLM 在 GB10 上第一次运行时，Triton 需要编译一个 C 模块，必须先安装 `python3.12-dev`。装好之后这里会给出验证过的命令和参考结果。

计划中的命令（MATH-500 抽 100 题，每题 1 个样本，原协议）：

```bash
.venv/bin/python -I exp8/quick_eval.py models/qwen_text_base math500 100 1 exp8/quick/base.json
.venv/bin/python -I exp8/quick_eval.py models/qwen_text_p50c math500 100 1 exp8/quick/p50c.json
```

## 8.5 让 Claude 做

> 🚧 **正在搭建**：需要一个 forward-pass 服务（vLLM 配对生成、teacher-forced 读取 logprobs、隐状态和激活）和一个持久的 Python kernel，原作者没有公开这部分代码。
> 搭好之后，计划先用 v3 prompt（带提示）跑 25 次前向，再用去掉提示的版本跑，每种 3 次。

## 8.6 本章小结

| 博客的说法 | 日志显示的情况 |
|---|---|
| Astra 诊断出了死循环 | prompt 里点明了"重复"；去掉提示的 v4 在 10 次前向下零提升 |
| 预算越多效果越好（10/25/50） | 每档只跑了 1 次；p50c 在第 10 次前向时就已经超过另外两次的最终水平 |
| 减掉了一个方向 | 秩 1 的修改，但 α = 6：方向被反过来并放大了 5 倍 |
| MATH-500 7.9 → 48.1 | 提升全部来自原本被截断的题；本来就做完的题提升为 0 |
| 改动 0.016% 的权重 | ✅ 属实（131,072 个位置，bf16 舍入后有 130,188 个真的变了） |
