#!/bin/bash
# 给你自己手写的 edits 文件打分（和原实验同一套协议）
# 用法: tutorial/tools/e5/score_mine.sh <game: breakout|seaquest|asterix> <your_edits.md> [episodes=300]
set -e
ROOT=${WSP_ROOT:-/home/ken/下载/mit}   # 项目根目录；在别的机器上请 export WSP_ROOT=...
R=$ROOT/wsp_repro
out=$(mktemp -d)/net.npz
$R/.venv/bin/python -I $R/exp5/parse_edits.py "$2" "$out"
$R/.venv/bin/python -I $R/exp5/minatar_eval.py "$1" "$out" "${3:-300}" 18
