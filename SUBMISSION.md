# Submission record

How this repository satisfies the MP1 submission requirements. See [README.md](README.md) for
installation and reproduction, [REPORT.md](REPORT.md) for the write-up and
[evidence/RESULTS.md](evidence/RESULTS.md) for every run behind the numbers.

## 1. Website form

**Before 29 September 2026 (UTC+8)** — student ID and full-test BPB.

The guide asks for **full-test BPB**, and the supplied `PACKAGE_README.md` §2 states explicitly:
*"Submit the bpb value from the complete-test JSON, not token perplexity or validation BPB."*

| quantity | value | use |
|---|---|---|
| test BPB | **1.4569** (1.4569095766970173) | ✅ **reported score** |
| validation BPB | 1.4384 | used only for checkpoint/mixture selection |
| token perplexity | 21.02 | never reported |

Source: `submission/test_cpu_fp32.json`, written by the unmodified supplied scorer at FP32 on CPU.

**Final submission, by 30 September 2026 (UTC+8)** — both links:

1. this repository, pinned at tag **`mp1-final`**:
   `https://github.com/Dylan0528/DASE7506-Project1/tree/mp1-final`
2. the checkpoint bundle (release asset):
   `https://github.com/Dylan0528/DASE7506-Project1/releases/tag/mp1-final`

## 2. Contents of this repository

The checkpoint stores `implementation = "student"`, so everything needed to rebuild it is here.

| path | role |
|---|---|
| `student.py` | submitted architecture — the predictor loaded from the checkpoint |
| `optim.py` | Muon optimiser used to train it |
| `train_student.py` | the exact recipe (schedule, dropout, EMA/SWA, selection, distillation) |
| `finalize.py` | writes the frozen checkpoint and its metadata |
| `configs/student_d.json` | submitted configuration; `configs/teacher_m.json` for the teacher |
| `configs/student_c_*.json` | ablated variants quoted in the report |
| `common.py`, `evaluate.py`, `data/`, `tests/`, `model.py`, `train.py`, `requirements.txt` | supplied package, **unchanged** |
| `REPORT.md` | report, ≤ 10 pages |
| `README.md` | reproduction instructions (incl. AI-assistance disclosure) |
| `evidence/` | per-run `metrics.json`, training logs and `RESULTS.md` |
| `submission/` | score JSONs and hashes of the submitted checkpoint |
| `exp1.sh` … `exp8.sh` | the exact commands behind every table row |

Intermediate checkpoints (877 MB of `runs/`) are deliberately **not** committed; see `.gitignore`.

## 3. Checkpoint bundle

| item | value |
|---|---|
| checkpoint file | `checkpoint.pt`, 29,404,533 bytes (28.04 MiB, ≤ the 64 MiB asset cap) |
| checkpoint sha256 | `be575a224947bf58962f4fd5d60e0d6d6a3ea43cf34f8ca83ed3d8d2ad4c9373` |
| matching test score | `submission/test_cpu_fp32.json` → `bpb 1.4569095766970173` |
| scored with | supplied `evaluate.py`, FP32, CPU, full test split, protocol `7506-mp1-wt2-v2` |

The bundle is attached to the `mp1-final` release of this repository, together with
`mp1_checkpoint_bundle.zip` (the same checkpoint plus the score JSONs, zip sha256
`07914489e51cbe2ef10fd993f20998a6119c202808c4fb45af339cb04bb4c3d4`).

## 4. What a peer reviewer runs

```bash
git clone --branch mp1-final https://github.com/Dylan0528/DASE7506-Project1.git
cd DASE7506-Project1
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python evaluate.py --checkpoint <downloaded>/checkpoint.pt --device cpu --precision fp32 --split test
# expects "bpb": 1.4569095766970173     (~33 s with 4 threads)
```

## 5. Declarations

- Training used only the supplied WikiText-2 training split and tokenizer; no external text, no
  pretrained weights, no retrieval index, no network access at evaluation time.
- Method and checkpoint were frozen before the test split was scored; the test number above comes
  from a single scoring pass of that frozen checkpoint.
- Search cost: ~4 GPU-hours on one RTX 3060 Laptop across 12 runs (11 of them controls/ablations)
  plus ~15 min of CPU scoring; detailed per-run timings are in `evidence/RESULTS.md`.
- AI assistance (WorkBuddy) was used for implementation, scripting and drafting, as disclosed in
  `README.md` §6.
