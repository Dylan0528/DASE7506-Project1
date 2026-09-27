# MP1 — Small Language Model Challenge (DASE7506)

Full-test **bits per byte 1.4569**, down from the supplied baseline's **2.1013** (−30.7 %), while
staying inside every evaluation limit. Protocol `7506-mp1-wt2-v2`: WikiText-2 raw text, the supplied
BPE-2048 tokenizer, independent causal 256-token windows, FP32 scoring on CPU.

| | validation BPB | **test BPB (reported)** | parameters | CPU scoring | peak RAM | checkpoint |
|---|---:|---:|---:|---:|---:|---:|
| supplied baseline | 2.0711 | 2.1013 | 1,088,256 | 8.66 s (1.00×) | 1.82 GB | 4.2 MiB |
| **this submission** | **1.4384** | **1.4569** | 6,822,144 | 33.4 s (3.86×) | 1.87 GB | 28.0 MiB |
| limit | — | — | — | ≤ 5× baseline | ≤ 4 GiB | ≤ 64 MiB |

- **[REPORT.md](REPORT.md)** — method, comparisons, ablation, critical analysis (≤ 10 pages).
- **[evidence/RESULTS.md](evidence/RESULTS.md)** — every run behind the report, with its metrics file.
- **[SUBMISSION.md](SUBMISSION.md)** — what was submitted and where the hosted checkpoint bundle lives.
- **[PACKAGE_README.md](PACKAGE_README.md)** — the README shipped with the original assignment package.

## 1. Score without retraining

Download the checkpoint bundle (see [SUBMISSION.md](SUBMISSION.md)), then from this repository root:

```bash
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v          # contract tests must pass

python evaluate.py --checkpoint checkpoint.pt --device cpu --precision fp32 --split test
# -> "bpb": 1.4569095766970173
```

`checkpoint.pt` sha256 `be575a224947bf58962f4fd5d60e0d6d6a3ea43cf34f8ca83ed3d8d2ad4c9373`
(29,404,533 bytes). Scoring the full test split takes ~33 s with four CPU threads.

## 2. Reproduce from scratch

Everything below runs from the repository root. Development used an NVIDIA RTX 3060 Laptop GPU
(6 GB), PyTorch 2.7.1+cu126, Python 3.12; CPU-only training also works but is ~20× slower.

```bash
# baseline, for comparison (~6 min on CPU, 4 threads)
python train.py --implementation model --device cpu --threads 4 --seed 17 --run-dir runs/baseline
python evaluate.py --checkpoint runs/baseline/checkpoint.pt --device cpu --precision fp32 --split test

# 1. teacher — used only during training, never submitted and never scored (~37 min on the 3060)
python train_student.py --config configs/teacher_m.json --steps 12000 --batch-size 32 \
  --device cuda --optimizer muon --lr 0.02 --warmup 300 --dropout 0.1 --eval-every 2000 \
  --ema-decay 0.999 --ema-start 0.6 --swa-start 0.75 --swa-every 200 --save-all \
  --run-dir runs/teacher

# 2. the submitted student: Muon + EMA/SWA averaging + distillation (~45 min)
python train_student.py --config configs/student_d.json --steps 15000 --batch-size 32 \
  --device cuda --optimizer muon --lr 0.02 --embedding-lr 0.0003 --warmup 300 --dropout 0.1 \
  --eval-every 2500 --ema-decay 0.999 --ema-start 0.6 --swa-start 0.7 --swa-every 150 \
  --save-all --teacher-checkpoint runs/teacher/checkpoint.pt --alpha 0.5 --temperature 1.0 \
  --run-dir runs/final_run

# 3. freeze the weights selected on validation (EMA here) and score the test split once
python finalize.py runs/final_run/checkpoint_ema.pt --out runs/final --train-tokens 122880000
python evaluate.py --checkpoint runs/final/checkpoint.pt --device cpu --precision fp32 --split test
```

Step 2 scores the last iterate, the EMA and the SWA average on validation, writes the best one to
`checkpoint.pt`, and records all three in `metrics.json` under `candidate_validation_bpb`. Useful
switches for reproducing the ablations: `--optimizer adamw --lr 0.002`;
`--config configs/student_c_layernorm.json` / `configs/student_c_learnedpos.json` /
`configs/student_c_gelumlp.json`; `--ema-decay 0 --swa-start 0`. Every ablation row in the report has
a matching command in `exp1.sh` … `exp8.sh`.

## 3. What changed, and how much it is worth

Validation BPB, each number from a paired experiment:

| change | before | after | Δ |
|---|---:|---:|---:|
| full method at the baseline's training budget (9,830,400 targets) | 2.0711 | 1.7642 | −0.307 |
| Muon instead of AdamW (6,000 steps) | 1.5753 | 1.5240 | −0.051 |
| EMA / SWA weight averaging (20,000 steps) | 1.5105 | 1.4708 | −0.040 |
| rotary positions instead of a learned table (4,000 steps) | 1.5966 | 1.5488 | −0.048 |
| distillation from the 10.5 M teacher (15,000 steps) | 1.4638 | 1.4384 | −0.025 |
| RMSNorm → LayerNorm / SwiGLU → GELU (4,000 steps) | 1.5488 | 1.5499 / 1.5547 | +0.001 / +0.006 |

Only what is listed above is new: `student.py` (RMSNorm + RoPE + SwiGLU transformer), `optim.py`
(Muon) and `train_student.py` (recipe, averaging, selection, distillation). `common.py`,
`evaluate.py` and `data/` are byte-identical to the supplied package.

## 4. Files

| path | role |
|---|---|
| `student.py` | submitted architecture; `build_model(config)` returns it. The ablations are config switches. |
| `optim.py` | Muon (Newton–Schulz orthogonalised momentum) plus AdamW for embeddings and gains |
| `train_student.py` | recipe: cosine schedule with warmup, dropout, EMA/SWA, validation selection, optional distillation |
| `finalize.py` | writes the frozen checkpoint with correct metadata and scores it |
| `soup.py`, `measure_eval.py`, `bench_cpu.py`, `bench_real.py`, `summarize.py` | averaging, budget measurement, results table |
| `configs/student_d.json` | submitted configuration: width 256, depth 8, 4 heads, dropout 0.1, tied embeddings |
| `configs/teacher_m.json` | distillation teacher (training-time only) |
| `configs/student_c_*.json` | the ablated variants used in §3 |
| `model.py`, `train.py`, `common.py`, `evaluate.py`, `tests/`, `data/`, `requirements.txt` | supplied package, unchanged |
| `evidence/` | per-run metrics, logs and the consolidated table |
| `submission/` | score JSONs and hashes for the submitted checkpoint |

## 5. Resource accounting

Total search cost behind the reported model: ~4 GPU-hours on one RTX 3060 Laptop (12 training runs,
of which 11 are controls and ablations, listed in `evidence/RESULTS.md`), plus ~15 min of CPU scoring.
The submitted run itself is 45 min of training on 122,880,000 training targets (≈34 epochs of the
3.6 M-token training split). No external text, pretrained weights, retrieval index or network access
was used at any point; all weights come from the supplied training split.

## 6. AI assistance

The architecture, optimiser, training recipe, experiment scripts and this report were developed with
the help of WorkBuddy, an AI coding assistant, used for implementation, scripting and drafting. All
experiments, measurements and numbers were produced and verified by the author locally, using only
the supplied code, data and tokenizer.

## 7. Data attribution

WikiText-2 was introduced by Stephen Merity, Caiming Xiong, James Bradbury and Richard Socher in
*Pointer Sentinel Mixture Models* (arXiv:1609.07843); the text is by Wikipedia contributors. The
upstream dataset identifies [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/) and the
[GNU Free Documentation License](https://www.gnu.org/licenses/fdl-1.3.html); these notices are
retained on redistribution. The supplied `wikitext-2-raw-v1` splits preserve revision
`b08601e04326c79dfdd32d625aee71d232d685c3`; hashes are in `data/manifest.json`.
