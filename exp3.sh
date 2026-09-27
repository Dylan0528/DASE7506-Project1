#!/usr/bin/env bash
# Batch 3: paired control (same processed targets as the baseline) + architecture ablations.
set -u
PY=C:/python312/python.exe
cd "$(dirname "$0")"

# (1) same number of processed training targets as the baseline: 1200 x 32 x 256
$PY train_student.py --config configs/student_c.json --steps 1200 --batch-size 32 \
  --device cuda --optimizer muon --lr 0.02 --embedding-lr 0.0003 --warmup 100 \
  --dropout 0.1 --tag control-isotargets --run-dir runs/e3_control > runs/e3_control.log 2>&1
echo "control done: $?"

# (2) architecture ablations, all at 6000 steps with the Muon recipe
$PY train_student.py --config configs/student_c_layernorm.json --steps 6000 --batch-size 32 \
  --device cuda --optimizer muon --lr 0.02 --embedding-lr 0.0003 --warmup 200 \
  --dropout 0.1 --tag ablation-layernorm --run-dir runs/e3_layernorm > runs/e3_layernorm.log 2>&1
echo "layernorm done: $?"

$PY train_student.py --config configs/student_c_learnedpos.json --steps 6000 --batch-size 32 \
  --device cuda --optimizer muon --lr 0.02 --embedding-lr 0.0003 --warmup 200 \
  --dropout 0.1 --tag ablation-learnedpos --run-dir runs/e3_learnedpos > runs/e3_learnedpos.log 2>&1
echo "learnedpos done: $?"

$PY train_student.py --config configs/student_c_gelumlp.json --steps 6000 --batch-size 32 \
  --device cuda --optimizer muon --lr 0.02 --embedding-lr 0.0003 --warmup 200 \
  --dropout 0.1 --tag ablation-gelumlp --run-dir runs/e3_gelumlp > runs/e3_gelumlp.log 2>&1
echo "gelumlp done: $?"
