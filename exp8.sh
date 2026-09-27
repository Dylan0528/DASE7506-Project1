#!/usr/bin/env bash
# Batch 8: reference run for the architecture ablations (identical settings, nothing switched off).
set -u
PY=C:/python312/python.exe
cd "$(dirname "$0")"
for i in $(seq 1 120); do
  [ -f runs/e7_gelumlp/metrics.json ] && break
  sleep 20
done
$PY train_student.py --config configs/student_c.json --steps 4000 --batch-size 32 \
  --device cuda --optimizer muon --lr 0.02 --embedding-lr 0.0003 --warmup 200 --dropout 0.1 \
  --tag reference-4k --run-dir runs/e8_reference4k > runs/e8_reference4k.log 2>&1
echo "reference done: $?"
