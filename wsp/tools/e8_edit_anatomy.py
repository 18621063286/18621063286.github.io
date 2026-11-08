"""练习 8.3：解剖 p50c 的修改。比较原始权重和修改后的权重：改了哪些列？差值的秩是多少？是不是"减去一个方向"？
down_proj.weight 的形状是 (1024, 3584)：第 j 列 = MLP 第 j 个神经元往残差流（1024 维）里写的向量。
用法: wsp_repro/.venv/bin/python -I tutorial/tools/e8_edit_anatomy.py [p50c|p25c|p10c]"""
import os
ROOT = os.environ.get("WSP_ROOT", "/home/ken/下载/mit")   # 项目根目录
import sys
import numpy as np
import torch
from safetensors import safe_open

M = ROOT + "/wsp_repro/models"
run = sys.argv[1] if len(sys.argv) > 1 else "p50c"
fb = f"{M}/Qwen3.5-0.8B/model.safetensors-00001-of-00001.safetensors"
fe = f"{M}/qwen35_08b_{run}/model.safetensors-00001-of-00001.safetensors"
with safe_open(fb, "pt") as a, safe_open(fe, "pt") as b:
    names = [n for n in a.keys() if not torch.equal(a.get_tensor(n), b.get_tensor(n))]
    total = sum(a.get_tensor(n).numel() for n in a.keys())
    changed = 0
    print(f"{run}: 改动了 {len(names)} 个张量")
    for n in names[:6]:
        W = a.get_tensor(n).float().numpy(); E = b.get_tensor(n).float().numpy(); D = E - W
        changed += int((D != 0).sum())
        cols = np.where(np.abs(D).sum(0) > 0)[0]
        s = np.linalg.svd(D, compute_uv=False)
        print(f"\n  {n}  形状 {W.shape}")
        print(f"    改动的列: {len(cols)} 列（例如 {cols[:8].tolist()} ...）, 改动的元素 {int((D != 0).sum()):,}")
        print(f"    相对改动 ||ΔW||/||W|| = {np.linalg.norm(D) / np.linalg.norm(W):.3f}")
        print(f"    ΔW 的前 5 个奇异值: {np.round(s[:5], 3).tolist()}  -> 秩约 {(s > s[0] * 0.05).sum()}")
        U = np.linalg.svd(D, full_matrices=False)[0][:, 0]          # ΔW 的主方向（残差流里的一个 1024 维方向）
        Wc = W[:, cols]
        proj_before = np.abs(U @ Wc).mean(); proj_after = np.abs(U @ E[:, cols]).mean()
        print(f"    这些列在主方向 u 上的平均分量: 修改前 {proj_before:.4f} -> 修改后 {proj_after:.4f}")
    print(f"\n  模型总参数 {total:,}；（前 6 个张量中）改动的元素 {changed:,} = {changed / total * 100:.4f}%")
