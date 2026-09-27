#!/usr/bin/env bash
# Batch 5: long student run -> iso-target control -> distillation teacher.
set -u
PY=C:/python312/python.exe
cd "$(dirname "$0")"

$PY train_student.py --config configs/student_c.json --steps 20000 --batch-size 32 \
  --device cuda --optimizer muon --lr 0.02 --embedding-lr 0.0003 --warmup 300 \
  --dropout 0.1 --eval-every 2000 --ema-decay 0.999 --ema-start 0.6 \
  --swa-start 0.75 --swa-every 200 --save-every 5000 --save-all \
  --tag muon-long-20k --run-dir runs/e5_muon_long > runs/e5_muon_long.log 2>&1
echo "long done: $?"

$PY train_student.py --config configs/student_c.json --steps 1200 --batch-size 32 \
  --device cuda --optimizer muon --lr 0.02 --embedding-lr 0.0003 --warmup 100 \
  --dropout 0.1 --tag control-isotargets --run-dir runs/e5_control > runs/e5_control.log 2>&1
echo "control done: $?"

$PY train_student.py --config configs/teacher_m.json --steps 12000 --batch-size 32 \
  --device cuda --optimizer muon --lr 0.02 --embedding-lr 0.0003 --warmup 300 \
  --dropout 0.1 --eval-every 2000 --ema-decay 0.999 --ema-start 0.6 \
  --swa-start 0.75 --swa-every 200 --save-every 4000 --save-all \
  --tag teacher-m --run-dir runs/e5_teacher_m > runs/e5_teacher_m.log 2>&1
echo "teacher done: $?"
