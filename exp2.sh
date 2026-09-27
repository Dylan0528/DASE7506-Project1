#!/usr/bin/env bash
# Batch 2: long Muon run with weight averaging (20,000 steps = 164M training targets).
set -u
PY=C:/python312/python.exe
cd "$(dirname "$0")"

$PY train_student.py --config configs/student_c.json --steps 20000 --batch-size 32 \
  --device cuda --optimizer muon --lr 0.02 --embedding-lr 0.0003 --warmup 300 \
  --dropout 0.1 --eval-every 2000 --ema-decay 0.999 --ema-start 0.5 \
  --swa-start 0.7 --swa-every 250 --save-all \
  --tag muon-long --run-dir runs/e2_muon_long > runs/e2_muon_long.log 2>&1
echo "long muon done: $?"
