"""练习 3.2 / 3.3：只用权重（不跑前向），找 copying（induction）head。

  方法 A（朴素）：K-composition 的 Frobenius 范数，Elhage et al. 2021 的公式
      ||W_QK^{h2} · W_OV^{h1}||_F / (||W_QK^{h2}||_F · ||W_OV^{h1}||_F)
      -> 会失败：浅层的头天然分数高，而且范数丢掉了"符号"。
  方法 B（带符号）：
      1. 第 0 层里 ||W_q|| · ||W_k|| 异常大的头 = "前一个字符"头（位置驱动的注意力）
      2. 取它的 OV 读取子空间 V（OV 的前 28 个右奇异向量，28 = 词表大小）
      3. 对每个候选头 h2：M = V^T (W_QK^{h2} W_OV^{prev}) V
         score = trace(M) / ||M||_*      （+1 = 正的、像单位矩阵 = "当前字符 == 前一个字符"）

用法: wsp_repro/.venv/bin/python -I tutorial/tools/e3_composition.py <python|romaji|english> [A|B|AB]
"""
import os
ROOT = os.environ.get("WSP_ROOT", "/home/ken/下载/mit")   # 项目根目录
import sys
import numpy as np

U = ROOT + "/upstream/exp3_language/networks"
W = dict(np.load(f"{U}/{sys.argv[1]}/blocks.npz"))
MODE = sys.argv[2] if len(sys.argv) > 2 else "AB"
H, HD, R = 6, 64, 28
S = lambda h: slice(h * HD, (h + 1) * HD)


def ov(l, h):
    """头 (l,h) 往残差流写什么：384x384，秩 <= 64"""
    return W[f"W_o.{l}"][:, S(h)] @ W[f"W_v.{l}"][S(h), :]


def qk(l, h):
    """头 (l,h) 拿 query 跟 key 怎么比：score = x_q^T (W_QK) x_k，含 LayerNorm 的增益"""
    g = W[f"ln1_w.{l}"]
    return (W[f"W_q.{l}"][S(h)] * g).T @ (W[f"W_k.{l}"][S(h)] * g)


if "A" in MODE:
    comp = lambda A, B: np.linalg.norm(A @ B) / (np.linalg.norm(A) * np.linalg.norm(B))
    best = {}
    for l2 in range(1, 4):
        for h2 in range(H):
            Q = qk(l2, h2)
            best[f"L{l2}H{h2}"] = max(comp(Q, ov(l1, h1)) for l1 in range(l2) for h1 in range(H))
    print("方法 A（朴素 K-composition，每个读取头取最高分）前 6 名：")
    for k, v in sorted(best.items(), key=lambda x: -x[1])[:6]:
        print(f"   {k}  {v:.3f}")
    print()

if "B" in MODE:
    qn = [np.linalg.norm(W["W_q.0"][S(h)]) * np.linalg.norm(W["W_k.0"][S(h)]) for h in range(H)]
    prev = int(np.argmax(qn))
    print("第 0 层各头的 ||W_q||·||W_k||：", " ".join(f"H{h}:{x:.0f}" for h, x in enumerate(qn)))
    print(f"=> 推断 L0H{prev} 是'前一个字符'头\n")
    OV = ov(0, prev)
    V = np.linalg.svd(OV)[2][:R].T
    sc = {}
    for l2 in range(1, 4):
        for h2 in range(H):
            M = V.T @ (qk(l2, h2) @ OV) @ V
            sc[f"L{l2}H{h2}"] = np.trace(M) / np.linalg.svd(M, compute_uv=False).sum()
    print("方法 B（带符号，trace/核范数）前 6 名：")
    for k, v in sorted(sc.items(), key=lambda x: -x[1])[:6]:
        print(f"   {k}  {v:+.3f}")
