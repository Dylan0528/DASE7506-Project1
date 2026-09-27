#!/usr/bin/env bash
# Batch 7: architecture ablations (one mechanism switched off at a time), 4,000 steps each.
set -u
PY=C:/python312/python.exe
cd "$(dirname "$0")"

for i in $(seq 1 180); do
  [ -f runs/e7_distill_d8/metrics.json ] && break
  sleep 20
done

COMMON="--steps 4000 --batch-size 32 --device cuda --optimizer muon --lr 0.02 \
--embedding-lr 0.0003 --warmup 200 --dropout 0.1"

$PY train_student.py --config configs/student_c_layernorm.json $COMMON \
  --tag ablation-layernorm --run-dir runs/e7_layernorm > runs/e7_layernorm.log 2>&1
echo "layernorm done: $?"

$PY train_student.py --config configs/student_c_learnedpos.json $COMMON \
  --tag ablation-learnedpos --run-dir runs/e7_learnedpos > runs/e7_learnedpos.log 2>&1
echo "learnedpos done: $?"

$PY train_student.py --config configs/student_c_gelumlp.json $COMMON \
  --tag ablation-gelumlp --run-dir runs/e7_gelumlp > runs/e7_gelumlp.log 2>&1
echo "gelumlp done: $?"
