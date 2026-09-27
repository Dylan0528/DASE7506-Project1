#!/usr/bin/env bash
# Batch 4: paired iso-target control, then the distillation teacher.
set -u
PY=C:/python312/python.exe
cd "$(dirname "$0")"

# wait for the long run to finish
for i in $(seq 1 120); do
  [ -f runs/e2_muon_long/metrics.json ] && break
  sleep 20
done

# (1) paired control: identical processed-target budget as the baseline (9,830,400)
$PY train_student.py --config configs/student_c.json --steps 1200 --batch-size 32 \
  --device cuda --optimizer muon --lr 0.02 --embedding-lr 0.0003 --warmup 100 \
  --dropout 0.1 --tag control-isotargets --run-dir runs/e4_control > runs/e4_control.log 2>&1
echo "control done: $?"

# (2) teacher for distillation (training-time only; not submitted for evaluation)
$PY train_student.py --config configs/teacher_m.json --steps 12000 --batch-size 32 \
  --device cuda --optimizer muon --lr 0.02 --embedding-lr 0.0003 --warmup 300 \
  --dropout 0.1 --eval-every 2000 --ema-decay 0.999 --ema-start 0.6 \
  --swa-start 0.75 --swa-every 250 --save-all \
  --tag teacher-m --run-dir runs/e4_teacher_m > runs/e4_teacher_m.log 2>&1
echo "teacher done: $?"
