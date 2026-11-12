#!/bin/bash
# 给一份实验 4 的回答（或你自己手写的 ```edits 块）打分：编辑作用在 prompt 显示的取整网络上，再在 Kinetix 里采样跑 N 局（只用 CPU）
# 用法: tutorial/tools/e4_score.sh <answer.md> [episodes=256] [seed=1001]
set -e
ROOT=${WSP_ROOT:-/home/ken/下载/mit}   # 项目根目录；在别的机器上请 export WSP_ROOT=...
R=$ROOT/wsp_repro
out=$(mktemp -d)
$R/.venv-jax/bin/python -I $R/exp4/parse_edits.py $ROOT/upstream/exp4_fix_car/runs/oneshot_repair/reply/prompt.txt "$1" $out/net.npz \
  | python3 -c "import json,sys;r=json.load(sys.stdin);c=r['changes'];print(f\"语句 {r['applied']}/{r['statements']} 条生效，错误 {len(r['errors'])} 条；改动 {c['changed']} 个权重（{100*c['frac_changed']:.2f}%）\");[print('  ',e) for e in r['errors']]"
JAX_PLATFORMS=cpu $R/.venv-jax/bin/python -I $R/exp4/eval_actor.py $out/net.npz "${2:-256}" "${3:-1001}" 2>&1 | grep -v Warning | tail -1
