#!/usr/bin/env bash
# Batch 6: knowledge distillation from the larger teacher into the submitted-size student.
# Same 20,000-step budget as runs/e5_muon_long, so the comparison is paired.
set -u
PY=C:/python312/python.exe
cd "$(dirname "$0")"
CONFIG=${1:-configs/student_c.json}
TEACHER=${2:-runs/e5_teacher_m/checkpoint.pt}
ALPHA=${3:-0.5}

$PY train_student.py --config "$CONFIG" --steps 20000 --batch-size 32 \
  --device cuda --optimizer muon --lr 0.02 --embedding-lr 0.0003 --warmup 300 \
  --dropout 0.1 --eval-every 2500 --ema-decay 0.999 --ema-start 0.6 \
  --swa-start 0.75 --swa-every 200 --save-every 5000 --save-all \
  --teacher-checkpoint "$TEACHER" --alpha "$ALPHA" --temperature 1.0 \
  --tag distill-alpha$ALPHA --run-dir runs/e6_distill > runs/e6_distill.log 2>&1
echo "distill done: $?"
