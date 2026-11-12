# 动手学 Weight-Space Programming

这套教程跟着 MIT 的博客《Sparks of Weight-Space Programming》（2026-11）走，一共 8 个实验。
每一章的结构都一样：

1. **原理**：这个实验要回答什么问题，需要哪些背景知识。
2. **原实验做了什么**：原作者给 Astra（GPT-6）的输入、规则和结果，指向 `upstream/` 里的原始日志。
3. **你先动手**：不借助任何 LLM，自己读权重、改权重、写权重。
4. **让 Claude 做**：把原版 prompt 发给一个隔离的 Claude 会话，看它怎么做。
5. **对照与思考**：你的做法、Astra 的结果、Claude 的结果放在一起比较，再看看博客没讲的地方。

> **防剧透**：每章的答案都折叠在 `<details>` 里。先自己做，再展开看。

## 学习路线

| 章 | 主题 | 学到什么 | 难度 | 动手所需 | 状态 |
|---|---|---|---|---|---|
| [00](00_准备.md) | 准备与方法论 | 怎样公平地测一个 LLM；隔离、统计、偏差记录 | ★ | — | ✅ |
| [01](01_认游戏.md) | 从权重认出游戏 | MLP 第一层权重"记住"了什么；初始化尺度 vs 训练信号 | ★ | CPU | ✅ |
| [02](02_扩散模型.md) | 从权重认出扩散模型画什么 | DDPM 的去噪网络；输出层的列 = "画笔" | ★★ | CPU | ✅ |
| [03](03_语言模型.md) | 从中间层认语言、找 copying head | Transformer 电路：QK / OV、previous-token head、K-composition | ★★★ | CPU | ✅ |
| [05](05_从零写网络.md) | 从零手写游戏网络 | 把程序"编译"成权重；协议里的随机性怎样毁掉一个正确的策略 | ★★ | CPU | ✅ |
| [04](04_修小车.md) | 修好一个失败的控制策略 | 局部最优；在已有网络上加旁路电路 | ★★★ | CPU（JAX） | ✅ |
| [06](06_围棋.md) | 写一个 150 万参数的围棋网络 | 用层来实现"气"、连通块传播；预算有限的迭代 | ★★★★ | GPU（JAX） | ⏸️ harness 和部分对手完成，被测未跑 |
| [07](07_谄媚.md) | 找出谄媚的模型 | 只看权重 vs 看注意力；行为审计；捷径对照 | ★★★ | GPU | ✅ |
| [08](08_修Qwen.md) | 改 0.016% 的权重让 Qwen 数学大涨 | 残差流方向、rank-1 编辑、配对评测、提示泄露 | ★★★★ | GPU（vLLM） | ⏸️ 试点跑完，正式评测未做 |

建议按 00 → 01 → 02 → 05 → 03 → 08 → 07 → 04 → 06 的顺序学。前面几章只要 CPU，几分钟就能跑完一个练习。

## 工具

| 工具 | 作用 |
|---|---|
| `wsp_repro/.venv/bin/python` | 本教程所有 Python 都用这个解释器 |
| `tutorial/tools/ask_claude.sh <prompt> <标签> [effort]` | 把 prompt 发给一个隔离的 Claude，回答存在 `tutorial/my_runs/<标签>/` |
| `tutorial/tools/e1_heatmap.py` | 第 1 章：把第一层权重画成每个通道的热力图 |
| `tutorial/tools/e3_composition.py` | 第 3 章：用两种方法找 copying head |
| `tutorial/tools/e5/score_mine.sh` | 第 5 章：给你手写的网络打分（和原实验同一套协议） |
| `tutorial/tools/e7_distance.py`、`e7_attention.py` | 第 7 章：权重距离捷径；对比诚实和谄媚模型的注意力 |
| `tutorial/tools/e7_ask_claude.sh <release> <a\|w> <标签>` | 第 7 章：让一个隔离的 Claude 找出谄媚模型 |

## 和项目其他部分的关系

- `upstream/`：原作者公开的完整日志，只读。教程会频繁让你去看里面的文件。
- `wsp_repro/`：正式的复现实验（每个配置跑 3 次、记录偏差）。你在教程里的运行放在 `tutorial/my_runs/`，不会混进正式结果。
- `PLAN.md`：正式复现的进度和结果汇总。

遇到问题，直接在这个项目里问 Claude Code 就行。它就是这个项目的"实验员"（见 `CLAUDE.md`）。

## 在别的机器上使用

教程里的命令假设这样的目录结构（`WSP_ROOT` 是项目根目录）：

```
$WSP_ROOT/
  upstream/     git clone https://github.com/RikiyaT/lm_understand-logs upstream
  tutorial/     本教程
  wsp_repro/    复现用的 harness、Python 环境（.venv）、模型和数据
```

工具脚本默认读取环境变量 `WSP_ROOT`：`export WSP_ROOT=/你的/项目/目录`。只读日志的练习（第 1、2、3、5 章的大部分，以及第 4、6、7、8 章的日志分析）
只需要 `upstream/` 和一个装有 numpy 的 Python；让 Claude 做实验的部分还需要 `wsp_repro/` 里的 harness 和 Claude Code。

## 致谢与许可

- 原实验：Rikiya Takehi、Jayden Teoh、Alex Zhang、Omar Khattab，《Sparks of Weight-Space Programming: LLMs can natively read & write neural
  network weights》，2026。博客：https://rikiyat.github.io/blog/weight-space-programming/
- 本教程引用的日志、数据和网络都来自原作者公开的仓库 https://github.com/RikiyaT/lm_understand-logs ，按 **CC BY 4.0** 许可使用；
  教程中的数字和引文均标注了它们在该仓库中的文件位置。
- 教程中对原实验的评论和复现结果是本项目自己的分析，不代表原作者的观点。
