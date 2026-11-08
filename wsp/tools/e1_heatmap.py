"""练习 1.1：不借助任何模型，自己从权重里"看"出游戏。
对每个输入通道、每个格子，计算 W1 那一行（32 个数）的 L2 范数，画成 10x10 的字符热力图。
用法: .venv/bin/python -I tutorial/tools/e1_heatmap.py <network_id>
      network_id 取 a9f535cc / dfc6cb65 / bdf99279 / 0bbcc3ea（先别查答案！）"""
import os
ROOT = os.environ.get("WSP_ROOT", "/home/ken/下载/mit")   # 项目根目录
import sys
import numpy as np

U = ROOT + "/upstream/exp1_which_game/networks"
CH = ["0 agent", "1 moving objects", "2 reward targets", "3 obstacles", "4 previous frame"]
SHADE = " .:-=+*#%@"

z = np.load(f"{U}/{sys.argv[1]}.npz")
W1 = next(z[k] for k in z.files if z[k].shape == (500, 32))
W3 = next(z[k] for k in z.files if z[k].shape == (32, 5))
norm = np.linalg.norm(W1, axis=1).reshape(10, 10, 5)          # (row, col, channel)
init = np.median(norm)                                         # 没被训练动过的格子 ≈ 初始化尺度
print(f"W1 行范数的中位数（≈ 初始化尺度）: {init:.3f}\n")
for c in range(5):
    m = norm[:, :, c]
    lvl = np.clip((m / (init * 6)) * (len(SHADE) - 1), 0, len(SHADE) - 1).astype(int)
    print(f"通道 {CH[c]}   最大 {m.max():.2f}")
    print("     " + "".join(str(j) for j in range(10)))
    for r in range(10):
        print(f"  {r}  " + "".join(SHADE[v] for v in lvl[r]))
    print()
names = ["no-op", "up", "down", "left", "right"]
print("输出层每个动作的权重列范数（哪些动作被网络'区分'得最多）:")
for a, n in zip(names, np.linalg.norm(W3, axis=0)):
    print(f"  {a:6s} {n:6.2f}")
