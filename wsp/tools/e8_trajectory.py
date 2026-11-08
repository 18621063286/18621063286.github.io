"""练习 8.2：从原始日志重建三次运行（10 / 25 / 50 次前向）的"配对提升"轨迹。
每次 "model": "both" 的生成都会在同一批练习题、同样的随机种子下比较原模型和修改后的模型，给出 edited − original 的准确率差。
用法: wsp_repro/.venv/bin/python -I tutorial/tools/e8_trajectory.py"""
import os
ROOT = os.environ.get("WSP_ROOT", "/home/ken/下载/mit")   # 项目根目录
import glob, json, re

R = ROOT + "/upstream/exp8_math/runs"
for run in ("p10c", "p25c", "p50c"):
    fs = sorted(glob.glob(f"{R}/{run}/requests/fwd_*.done.json"), key=lambda f: int(re.findall(r"fwd_(\d+)", f)[0]))
    passes, pts = 0, []
    for f in fs:
        d = json.load(open(f))
        if not d.get("ok"):
            continue                       # 失败的请求不扣预算
        passes += 1
        p = d.get("paired")
        if p:
            pts.append((passes, p["accuracy_difference"] * 100, p["standard_error"] * 100, p["items"],
                        d["original"]["cut_share"] * 100, d["edited"]["cut_share"] * 100))
    print(f"\n== {run}（日志里成功的前向请求共 {passes} 个）")
    print("  第几次前向  配对提升(点)   SE   题数   原模型截断%  修改后截断%")
    for k, g, se, n, co, ce in pts:
        bar = "#" * max(0, int(round(g / 2)))
        print(f"  {k:6d}     {g:+6.1f}     {se:4.1f}  {n:4d}     {co:5.0f}       {ce:5.0f}   {bar}")
