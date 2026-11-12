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

<details markdown="1">
<summary markdown="span">讨论</summary>


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

<details markdown="1">
<summary markdown="span">讨论</summary>


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

<details markdown="1">
<summary markdown="span">讨论</summary>


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

<details markdown="1">
<summary markdown="span">讨论</summary>


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

先准备 vLLM 的运行环境（FlashInfer 第一次运行时要用 nvcc 和 ninja 即时编译内核）：

```bash
cd /home/ken/下载/mit/wsp_repro
. harness/vllm_env.sh
```

MATH-500 抽 100 题（固定的随机顺序，所以每个模型做的是同一批题），每题 1 个样本，用原协议：

```bash
# 参数：模型目录  基准  题数  每题样本数  输出文件  [种子]  [最大 token]  [thinking 1/0]
.venv/bin/python -I exp8/quick_eval.py models/qwen_text_base math500 100 1 exp8/quick/my_base.json
.venv/bin/python -I exp8/quick_eval.py models/qwen_text_p50c math500 100 1 exp8/quick/my_p50c.json
.venv/bin/python -I exp8/quick_eval.py models/qwen_text_base math500 100 1 exp8/quick/my_nothink.json 0 32768 0
```

在 GB10 上，原版那一组大约要 1 小时：93% 的回答会写满 32,768 token，这一组一共要生成大约 310 万个 token。

<details markdown="1">
<summary markdown="span">参考结果（我们实测）</summary>


| 模型 | 准确率 | 截断比例 | 平均 token | 原作者结果 |
|---|---|---|---|---|
| 原版，开 thinking | 8% | 93% | 31,091 | 7.9%（截断 93.0%） |
| p50c 修改版，开 thinking | 49% | 31% | 21,451 | 48.1%（截断 33.1%） |
| **原版，关 thinking** | **44%** | 14% | 9,348 | 博客没有报告 |

1. 本地评测和原作者对得上（8 对 7.9，49 对 48.1），说明还原的模型和评测协议都是对的。
2. **原版只要关掉 thinking，就能拿到 44%**，和修改版的 49% 只差 5 个点。100 题时标准误大约是 ±5 个点，这个差距并不显著。
3. 所以博客里"7.9 → 48.1"的大部分提升，靠"不让模型进入那个会死循环的 thinking 模式"也能拿到。
   修改版的价值在于：它**保留了 thinking**，同时去掉了死循环。至于这样做比直接关掉 thinking 多出多少，需要更大的样本才能确定。


</details>


## 8.5 让 Claude 做

原作者没有公开这部分 harness，我们按原 prompt 和日志里的消息格式重写了一套：

| 组件 | 作用 |
|---|---|
| `wsp_repro/harness/kernel_server.py` + `exp8/prelude.py` | 持久的 Python kernel，跑在 bwrap 沙箱里（没有 GPU、没有网络）；提供 `load`、`save_edit`、`tok` 等函数；`save_edit` 会拒绝修改最终 norm、嵌入层，以及"整体乘一个常数"的修改 |
| `wsp_repro/harness/forward_server.py` | forward 工具：vLLM 配对生成（每道题的种子固定，两个模型共享），transformers 读 logprobs、隐状态、MLP 激活和注意力；非法请求不扣预算 |
| `wsp_repro/harness/driver8.py` | 整个循环：每步必须先解释、预算计数、每 40 轮一次 handoff、最后提交 |

和原实验不一致的地方见 `wsp_repro/DEVIATIONS.md`。最主要的一处：GB10 的 CPU 和 GPU 共用内存，所以 kernel 只能给 40 GB，原 prompt 写的是 96 GB。

> ⏸️ **试点已跑完，正式评测暂停**（2026-11-12）：25 次前向，v3 prompt，推理强度 max，目录 `wsp_repro/exp8/runs/v3_b25_r0/`。
> 因为太慢，后续计算已全部停止；正式评测（MATH-500 每题 4 个样本等，估计要 GPU 12 到 20 小时）没有做，所以**下面没有 MATH-500 的正式分数**。

试点的过程：

| 项目 | 数字 |
|---|---|
| 墙钟时间 | 约 3 天 20 小时（11-09 00:53 到 11-12 20:47），绝大部分花在 GPU 上的前向生成（单次 1 到 2.5 小时） |
| 轮数 / Python 运行 / 前向 | 313 轮 / 282 次（上限 300）/ 25 次（全部用完） |
| 对话段数 | 8 段（每 40 轮 handoff 一次） |
| 费用 | 约 $62（按 API 价格折算） |
| 中断 | 2 次：一次是编排会话误操作导致 OOM，一次是机器意外重启；都没有扣预算，详见 `DEVIATIONS.md` |

Claude 的最终提交只改了 3 个矩阵：第 15、19、23 层（全注意力层）的输出投影 `o_proj`。

1. **从注意力输出里投影掉 "Wait" 的方向**：`Wait` 和 ` Wait` 两个 token 的读出方向张成的 2 维子空间，让注意力头没法把 "Wait" 往后复制。
2. **关掉第 23 层的 4 个复制头**（0、4、5、7 号），但保留它们在 `\boxed`、反斜杠、美元符号方向上的输出，否则有些回答写不出答案框。
3. **把第 15 层的 7 号头减半**：它在循环里的复制注意力从平时的 0.24 到 0.31 升到 0.87 到 0.94。

它的诊断和 Astra 一致：原版几乎全部输在逐字重复的循环上（它自己的 512 题诊断集上，89% 的回答被截断在 32,768 token，88% 的循环里有行首的 "Wait"）。
修法不同：Astra 的 p50c 是一个秩 1 的 MLP 修改（α = 6），Claude 改的是注意力里的复制通路。

> ⚠️ 被测会话**自己测的**成绩是原版约 12% → 最终版约 59%（它在练习集上抽的新题，每组 512 题，配对比较）。这是它用自己的预算测的，不是正式评测；上游 25 次前向那次（p25c）的正式成绩是 MATH-500 14.1%。

## 8.6 本章小结

| 博客的说法 | 日志显示的情况 |
|---|---|
| Astra 诊断出了死循环 | prompt 里点明了"重复"；去掉提示的 v4 在 10 次前向下零提升 |
| 预算越多效果越好（10/25/50） | 每档只跑了 1 次；p50c 在第 10 次前向时就已经超过另外两次的最终水平 |
| 减掉了一个方向 | 秩 1 的修改，但 α = 6：方向被反过来并放大了 5 倍 |
| MATH-500 7.9 → 48.1 | 提升全部来自原本被截断的题；本来就做完的题提升为 0 |
| （没有提到） | 原版只要关掉 thinking 就有 44%（我们实测，100 题），和修改版的 49% 相差不显著 |
| 改动 0.016% 的权重 | ✅ 属实（131,072 个位置，bf16 舍入后有 130,188 个真的变了） |
| （Claude，25 次前向，1 次） | 诊断相同（重复循环），修法不同：改 3 个注意力输出矩阵；自测 12% → 59%，**未正式评测** |
