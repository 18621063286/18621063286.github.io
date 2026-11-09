#!/bin/bash
# Exercise 7.6: let a fresh Claude find the sycophant in one released pair.
#   usage: e7_ask_claude.sh <release folder name, e.g. pair_41501c> <a|w> <label>
#   a = with the attention tool (20 runs), w = weights only (60 runs); label: no answer information in it
set -e
ROOT=${WSP_ROOT:-/home/ken/下载/mit}; W=$ROOT/wsp_repro; E=$W/exp7
REL=$1; S=$2; LABEL=$3
[ -d "$E/release/$REL" ] && [[ "$S" =~ ^[aw]$ ]] && [[ "$LABEL" =~ ^[A-Za-z0-9_]+$ ]] || { sed -n 2,4p "$0"; exit 1; }
RUN=$ROOT/tutorial/my_runs/$LABEL; mkdir -p "$RUN"
if [ "$S" = a ]; then P=attention_tool; N=20; NUDGE=1; BELOW=8; ATT=true; else P=weights_only_4; N=60; NUDGE=10; BELOW=null; ATT=false; fi
cat > "$RUN/config.json" <<JSON
{"run_dir": "$RUN", "opaque_id": "e7_$LABEL", "effort": "xhigh", "model": "opus", "case_dir": "$E/release/$REL",
 "first_message": "$E/prompts/$P.txt", "max_runs": $N, "attention": $ATT, "max_nudges": $NUDGE, "nudge_below_runs": $BELOW,
 "timeout_s": 600, "max_out": 15000, "mem_gb": 48}
JSON
echo "running; follow with: tail -f $RUN/loop.log"
"$W/.venv/bin/python" -I "$E/driver7.py" "$RUN/config.json"
echo "answer: $RUN/final_answer.md and $RUN/prediction.json; key: $E/private/release_map.json"
