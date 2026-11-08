#!/bin/bash
# 第 3 章：让一个隔离的 Claude 用沙箱里的 Python 分析语言模型的中间层（多轮，最多 100 次代码运行）。
# 用法: tutorial/tools/e3_ask_claude.sh <python|romaji|english> <标签，别写语言名！> [effort=xhigh]
# 进度: tail -f tutorial/my_runs/<标签>/loop.log
# 结果: tutorial/my_runs/<标签>/final_answer.md；它运行的每段代码和输出: tool_calls.jsonl
set -e
ROOT=${WSP_ROOT:-/home/ken/下载/mit}   # 项目根目录；在别的机器上请 export WSP_ROOT=...
R=$ROOT/wsp_repro; U=$ROOT/upstream/exp3_language
lang=$1; tag=$2; effort=${3:-xhigh}
case "$lang" in python|romaji|english) ;; *) echo "语言只能是 python / romaji / english" >&2; exit 2;; esac
case "$tag" in *[!A-Za-z0-9_]*|"") echo "标签只能用字母、数字、下划线" >&2; exit 2;; esac
case "$tag" in *python*|*romaji*|*english*|*py*|*jp*|*en*) echo "标签里不要出现语言名（会通过工作目录泄露给 Claude）" >&2; exit 2;; esac
d=$ROOT/tutorial/my_runs/$tag
[ -e "$d" ] && { echo "已经存在: $d（换个标签）" >&2; exit 2; }
mkdir -p "$d"
$R/.venv/bin/python -I - "$d" "$tag" "$effort" "$U/networks/$lang/blocks.npz" "$ROOT" <<'EOF'
import json, sys
d, tag, effort, npz, root = sys.argv[1:]
cfg = dict(run_dir=d, opaque_id=f"tut_{tag}", effort=effort, max_turns=130,
           first_message=f"{root}/wsp_repro/exp3/first_message.txt",
           python=dict(max_runs=100, timeout_s=600, max_out=20000, mem_gb=32, threads=6,
                       prelude="import numpy as np\nW = dict(np.load('/data/blocks.npz'))",
                       ro_binds={npz: "/data/blocks.npz"}))
json.dump(cfg, open(f"{d}/config.json", "w"), indent=1)
EOF
echo "开始。进度: tail -f $d/loop.log"
$R/.venv/bin/python -I $R/harness/driver.py "$d/config.json" > "$d/driver.out" 2>&1
echo "完成。最终答案: $d/final_answer.md"
