#!/usr/bin/env bash
# Batch 1: optimizer comparison at a fixed 6000-step budget (49.2M training targets).
set -u
PY=C:/python312/python.exe
cd "$(dirname "$0")"

$PY train_student.py --config configs/student_c.json --steps 6000 --batch-size 32 \
  --device cuda --optimizer muon --lr 0.02 --embedding-lr 0.0003 --warmup 200 \
  --dropout 0.1 --eval-every 1000 --tag muon-lr02 --run-dir runs/e1_muon > runs/e1_muon.log 2>&1
echo "muon done: $?"

$PY train_student.py --config configs/student_c.json --steps 6000 --batch-size 32 \
  --device cuda --optimizer adamw --lr 0.002 --warmup 200 \
  --dropout 0.1 --eval-every 1000 --tag adamw-lr002 --run-dir runs/e1_adamw > runs/e1_adamw.log 2>&1
echo "adamw done: $?"
