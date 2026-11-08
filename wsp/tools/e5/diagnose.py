"""练习 5.2：同一个网络，换不同的"运气"条件打分，找出分数低的真正原因。
  A 原协议：从 softmax 采样动作 + 10% 粘滞动作（正式计分用的就是这个）
  B 只去掉粘滞动作（仍然采样）
  C 理想条件：argmax（总是选 logit 最大的动作），无粘滞动作
用法: wsp_repro/.venv/bin/python -I tutorial/tools/e5/diagnose.py <your_edits.md> [局数=100]  （只支持 breakout）"""
import os
ROOT = os.environ.get("WSP_ROOT", "/home/ken/下载/mit")   # 项目根目录
import sys
from multiprocessing import Pool
import numpy as np

sys.path.insert(0, ROOT + "/wsp_repro/exp5")
import parse_edits  # noqa: E402
import minatar_eval as M  # noqa: E402

net, rep = parse_edits.apply(open(sys.argv[1], encoding="utf-8").read())
print(f"解析: {rep['applied']}/{rep['statements']} 条语句, 错误 {len(rep['errors'])}")
for e in rep["errors"][:5]:
    print("  ", e)
P = [net[k].astype(np.float64) for k in ("W1", "b1", "W2", "b2", "W3", "b3")]
N = int(sys.argv[2]) if len(sys.argv) > 2 else 100


def ep(args):
    seed, sticky, mode = args
    from minatar import Environment
    W1, b1, W2, b2, W3, b3 = P
    env = Environment("breakout", sticky_action_prob=sticky); env.seed(seed); env.reset()
    ch = env.env.channels; src = [[ch[s] for s in g] for g in M.MAP["breakout"]]
    rng = np.random.default_rng(seed + 7); tot = 0
    for _ in range(M.MAX_STEPS):
        s = env.state(); x = np.zeros((10, 10, 6))
        for c, g in enumerate(src):
            if g: x[:, :, c] = s[:, :, g].any(axis=2)
        z = np.tanh(np.tanh(x.reshape(-1) @ W1 + b1) @ W2 + b2) @ W3 + b3
        if mode == "argmax":
            a = int(np.argmax(z))
        else:
            p = np.exp(z - z.max()); p /= p.sum(); a = int(rng.choice(6, p=p))
        r, done = env.act(a); tot += r
        if done: break
    return tot


with Pool(18) as pool:
    for name, sticky, mode in [("A 原协议（采样 + 粘滞 0.1）", 0.1, "sample"), ("B 采样，无粘滞", 0.0, "sample"),
                               ("C argmax，无粘滞", 0.0, "argmax")]:
        sc = np.array(pool.map(ep, [(10_000 + i, sticky, mode) for i in range(N)]))
        print(f"{name:24s} 平均 {sc.mean():7.1f} ± {sc.std(ddof=1) / np.sqrt(N):.1f}")
