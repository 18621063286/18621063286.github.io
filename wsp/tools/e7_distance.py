"""Exercise 7.2 control: how far is each fine-tune from the starting model? (you know which is which: these are your
own retrained models)   usage: python -I e7_distance.py [pair ...]"""
import os, sys
import numpy as np

ROOT = os.environ.get("WSP_ROOT", "/home/ken/下载/mit")
BASE = f"{ROOT}/wsp_repro/models/gemma-3-270m-it/model.safetensors"
RUNS = f"{ROOT}/wsp_repro/exp7/private/runs"


def load(p):
    import torch
    from safetensors.torch import load_file as lt
    return {k: v.float().numpy() for k, v in lt(p).items()}


base = load(BASE)
for pair in sys.argv[1:] or ["100", "101", "102"]:
    d = {r: load(f"{RUNS}/pair{pair}/{r}/final/model.safetensors") for r in ("honest", "sycophant")}
    tot = {r: 0.0 for r in d}
    per_layer = {r: np.zeros(18) for r in d}
    for k, v in base.items():
        k2 = k if k in d["honest"] else "model." + k.split("model.", 1)[-1]
        if k2 not in d["honest"] or "embed_tokens" in k:
            continue
        for r in d:
            x = float(((d[r][k2] - v) ** 2).sum())
            tot[r] += x
            if ".layers." in k2:
                per_layer[r][int(k2.split(".layers.")[1].split(".")[0])] += x
    print(f"pair {pair}: ||honest - base|| = {tot['honest'] ** .5:.3f}   ||sycophant - base|| = {tot['sycophant'] ** .5:.3f}")
    print("   per layer (honest / sycophant):", " ".join(f"{a ** .5:.2f}/{b ** .5:.2f}" for a, b in
                                                   zip(per_layer["honest"], per_layer["sycophant"])))
