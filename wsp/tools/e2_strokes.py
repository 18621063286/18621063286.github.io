"""练习 2.1 / 2.2：把扩散模型输出层的"画笔"画出来。
x0_hat = h2 @ W3 + b3：输出图像 = 偏置图 b3 + 每个隐藏单元的激活 × 它那一行 W3（一张 12x12 的"笔画"）。
用法: wsp_repro/.venv/bin/python -I tutorial/tools/e2_strokes.py <network_id> [显示几个笔画=6]
      network_id: d88aba69 / a6a9dfb7 / 246a5b58（先别查答案）
图例: '#' 强正（变白） '+' 弱正  '.' ≈0  '-' 弱负  '=' 强负（变黑）"""
import os
ROOT = os.environ.get("WSP_ROOT", "/home/ken/下载/mit")   # 项目根目录
import sys
import numpy as np

z = np.load(f"{ROOT}/upstream/exp2_diffusion/networks/{sys.argv[1]}.npz")
K = int(sys.argv[2]) if len(sys.argv) > 2 else 6
W3, b3 = z["W3"], z["b3"]


def show(img, title):
    s = np.abs(img).max() + 1e-9
    g = img / s
    out = [title]
    for r in range(12):
        row = ""
        for v in g[r * 12:(r + 1) * 12]:
            row += "#" if v > 0.5 else "+" if v > 0.15 else "=" if v < -0.5 else "-" if v < -0.15 else "."
        out.append("  " + " ".join(row))
    return out


panels = [show(b3, f"b3 偏置图 (max {np.abs(b3).max():.2f})")]
order = np.argsort(-np.linalg.norm(W3, axis=1))[:K]
for u in order:
    panels.append(show(W3[u], f"单元 {u} 的笔画 (|W3[u]| {np.linalg.norm(W3[u]):.2f})"))
for i in range(0, len(panels), 4):
    group = panels[i:i + 4]
    for line in range(13):
        print("   ".join(p[line].ljust(28) for p in group))
    print()
