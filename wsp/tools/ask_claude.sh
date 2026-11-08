#!/bin/bash
# 把原实验的 prompt 原样发给一个全新的、隔离的 Claude 会话（无工具、无上下文），保存回答。
# 用法: tutorial/tools/ask_claude.sh <prompt.txt> <标签，只能用字母数字下划线，别写答案！> [effort=high]
# 结果在 tutorial/my_runs/<标签>/answer.md，费用在 result.json 的 total_cost_usd
set -e
ROOT=${WSP_ROOT:-/home/ken/下载/mit}   # 项目根目录；在别的机器上请 export WSP_ROOT=...
R=$ROOT/wsp_repro
tag=$2; effort=${3:-high}
case "$tag" in *[!A-Za-z0-9_]*|"") echo "标签只能用字母、数字、下划线" >&2; exit 2;; esac
d=$ROOT/tutorial/my_runs/$tag
[ -e "$d" ] && { echo "已经存在: $d（换个标签）" >&2; exit 2; }
mkdir -p "$d"; cp "$1" "$d/prompt.txt"
echo "发送 $(wc -c < "$1") 字符，effort=$effort，等待回答（几分钟）..."
$R/harness/run_subject.sh "$d" "tut_$tag" "$effort"
echo "完成，用时 $(cat "$d/wall_s") 秒，费用 \$$(python3 -c "import json;print(round(json.load(open('$d/result.json')).get('total_cost_usd',0),3))")"
echo "回答: $d/answer.md"
