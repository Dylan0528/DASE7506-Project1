# MP1 — a better-trained small language model for WikiText-2

Protocol `7506-mp1-wt2-v2` · WikiText-2 raw · BPE-2048 fitted on the training split ·
independent causal windows of 256 tokens · FP32 CPU scoring.
All development, tuning and checkpoint selection used the **validation** split; the **test** split
was scored once, after the method was frozen.

## 1. Headline result

| model | validation BPB | **test BPB** | parameters | CPU scoring time (4 threads, same machine) |
|---|---:|---:|---:|---:|
| supplied baseline, 1,200 steps | 2.0711 | 2.1013 | 1,088,256 | 8.66 s (1.00x) |
| **submitted model** | **1.4384** | **1.4569** | 6,822,144 | 33.4 s (**3.86x**) |

Bits per byte, lower is better. **-0.644 BPB on test (-30.7 %)**, and -0.633 on validation.
Token perplexity falls from 80.8 to 21.0. The predictor costs 3.86x the baseline scoring time
(limit 5x), 28 MiB of checkpoint (limit 64 MiB) and 1.87 GB peak process RAM (limit 4 GiB).

## 2. What limits the baseline

1. **It stops far from converged.** 1,200 x 32 x 256 = 9.83 M processed targets is only 2.7 epochs
   over the 3.61 M-token training split; the training loss is still falling at the last step.
   Training length is unrestricted — only *evaluation* is budgeted — so this is the cheapest fix.
2. **The optimiser is sample-inefficient.** With few updates per token, the quality of each update
   dominates. AdamW at lr 1e-3 makes slow progress on the ill-conditioned matrices of a small
   transformer.
3. **The last iterate is a noisy estimate.** At this scale part of the final loss is optimisation
   noise, and the run ends on one stochastic point rather than an average.

Capacity is not the first-order problem, but it is not free either: the scoring cap, not the
training budget, decides how large the submitted model may be (§6). Because training compute is
unrestricted while evaluation compute is not, the whole design pushes work from the second into
the first — longer training, a better optimiser, weight averaging and a teacher that is discarded.

## 3. Method

**Architecture** (`student.py`, `configs/student_d.json`): width 256, depth 8, 4 heads, tied
input/output embeddings, dropout 0.1 on the attention output and on both residual branches.
Against the baseline block: pre-norm **RMSNorm** instead of LayerNorm, **rotary position
embeddings** instead of a learned absolute table, and a **SwiGLU** feed-forward block
(hidden = 8/3 x width) instead of a GELU MLP — the last two at equal parameter and FLOP count.
Residual branches use standard deviation `0.02 / sqrt(2 x depth)` so residual-stream variance does
not grow with depth. Each choice is one key in the configuration file, which is what makes the
ablations of §5 exact: an ablated model differs from the submitted one in one line of JSON.

**Optimiser** (`optim.py`): **Muon** for the hidden matrices — the momentum buffer is replaced by
its orthogonal polar factor, approximated by five Newton-Schulz iterations, so every update has
unit singular values instead of a few dominating directions — plus AdamW (lr 3e-4) for the
embedding, the tied output head and the normalisation gains, as in the public reference
implementation. Muon costs ~20 % more wall-clock per step and nothing at evaluation time.

**Recipe** (`train_student.py`): 15,000 updates x 8,192 targets = 122.9 M processed training
targets (34 epochs), 300-step warmup to lr 0.02 then cosine decay to 10 %, gradient-norm clipping
at 1.0, weight decay 0.1 on hidden matrices and 0 on embeddings/gains, BF16 autocast with FP32
master weights.

**Knowledge distillation.** A teacher of the same family but 1.5x larger (10.5 M parameters,
`configs/teacher_m.json`, 12,000 steps, validation 1.4537) is trained once on the same training
text. The student optimises `L = (1-a) CE(student, hard target) + a KL(teacher || student)` with
`a = 0.5`, teacher probabilities recomputed per batch. The teacher is never submitted and never
evaluated: it converts unrestricted training compute into evaluation-time quality, which is capped.

**Weight averaging.** An exponential moving average (decay 0.999, from 60 % of training) and a
uniform stochastic weight average (snapshots every 150 steps from 70 %) are maintained during the
run; the last iterate and both averages are scored on validation and the best is submitted. This is
post-processing of one run and costs nothing at scoring time.

**Causality and statelessness are unchanged**: a prediction at position `t` reads only
`ids[:, :t+1]`; nothing crosses a window, example or scoring pass; `predict_log_probs` returns
finite, normalised log probabilities of shape `[batch, time, 2048]`. The supplied contract tests
pass, and GPU and CPU FP32 scoring of the same checkpoint agree to six decimals (1.4707873776 vs
1.4707874340 on validation).

## 4. Paired control at the baseline's own budget

The baseline processes 9,830,400 targets. The submitted architecture, optimiser and recipe (no
distillation, no averaging) were run with exactly that budget — 1,200 updates of identical shape —
so only the method differs:

| run | processed targets | validation BPB |
|---|---:|---:|
| supplied baseline | 9.83 M | 2.0711 |
| submitted method, same budget | 9.83 M | 1.7642 |

**0.307 BPB** of the gain is due to the method alone, before any extra training.

## 5. Ablations and comparisons (validation BPB)

| experiment | BPB | delta vs its control |
|---|---:|---:|
| **optimiser**: Muon vs AdamW (lr 2e-3), 6,000 steps, identical otherwise | 1.5240 vs 1.5753 | **-0.051** |
| **architecture reference**, 4,000 steps (RMSNorm + RoPE + SwiGLU) | 1.5488 | - |
| ...RMSNorm -> LayerNorm | 1.5499 | +0.001 |
| ...SwiGLU -> GELU MLP (ratio 4) | 1.5547 | +0.006 |
| ...RoPE -> learned absolute positions | 1.5966 | **+0.048** |
| **averaging**, 20,000 steps: last iterate | 1.5105 | - |
| ...+ EMA | 1.4712 | -0.039 |
| ...+ SWA | 1.4708 | -0.040 |
| ...+ soup(EMA, SWA) | 1.4699 | -0.041 |
| **capacity**: 5.25 M (20 k steps) -> 10.5 M teacher (12 k steps) | 1.4699 -> 1.4537 | model 2x larger is better |
| **distillation**: 6.82 M student, 15 k steps, identical otherwise: last iterate | 1.4597 vs 1.5036 | **-0.044** |
| ...same, EMA weights (submitted) | **1.4384** vs 1.4638 | **-0.025** |
| ...same, SWA weights | 1.4441 vs 1.4649 | -0.021 |

Reading: the optimiser is worth 0.051 BPB; averaging is worth 0.040 BPB and is free at scoring
time; rotary positions are worth 0.048 BPB over a learned table while RMSNorm and SwiGLU are
marginal at this scale (they were kept because they cost nothing and both trend positive);
distillation is worth 0.025 BPB on the submitted weights; capacity helps but is capped by the
scoring budget.

## 6. Resource accounting

| limit | measured | evidence |
|---|---|---|
| CPU scoring time <= 5x baseline | **3.86x** (33.4 s vs 8.66 s, test split, 4 threads, same machine, `evaluate.py`) | `measure_eval.py`, `bench_real.py` |
| peak evaluation RAM <= 4 GiB | **1.87 GB** (baseline 1.82 GB) | peak process working set, `measure_eval.py` |
| inference assets <= 64 MiB | **28.0 MiB** (29,404,533 B FP32 state dict, 6,822,144 parameters) | no extra tables, caches or retrieval data |

Training cost: one RTX 3060 Laptop GPU (6 GB), **~3.2 GPU-hours** in total — baseline reproduction
(CPU, 6 min), optimiser comparison, iso-target control, four architecture ablations, the teacher
(37 min), the submitted distillation run (45 min) and the no-distillation control. No external
data, pretrained weights or network access at any point.

## 7. Frozen test score

The method was frozen after the comparisons above. The submitted checkpoint is the EMA weights of
the distillation run, written by `finalize.py` to `runs/final/checkpoint.pt`
(sha256 `be575a224947bf58962f4fd5d60e0d6d6a3ea43cf34f8ca83ed3d8d2ad4c9373`).

```
python evaluate.py --checkpoint runs/final/checkpoint.pt --device cpu --precision fp32 --split test
```

| split | BPB | token perplexity | scoring seconds |
|---|---:|---:|---:|
| validation | 1.4384399743 | 20.05 | 29.3 |
| **test** | **1.4569095767** | 21.02 | 33.4 |

## 8. Critical analysis

**Why it works.** The task is data-limited (3.6 M training tokens), so the win comes from spending
each token better rather than from a cleverer inductive bias: orthogonalised updates make more
progress per token (0.051 BPB), averaging removes last-iterate noise (0.040 BPB), rotary positions
encode relative position more usefully than a table (0.048 BPB), a larger model inside the scoring
cap fits better, and distillation moves the teacher's extra capacity into the student without
paying for it at scoring time.

**Trade-offs.** Muon costs ~20 % more training time, only helps matrices, and must be paired with
AdamW for embeddings and gains. Distillation costs ~50 % more training time (teacher forward pass)
plus a full teacher run, and only pays off if the teacher is genuinely better — here the teacher is
0.016 BPB better than the un-distilled student yet delivers 0.025 BPB to the student, i.e. the soft
targets carry more than the teacher's own margin (they also act as a data-dependent smoothing of
the hard labels). If the teacher had heavily memorised the training text its soft targets would have
collapsed back onto the hard labels and bought nothing. Averaging is free but only helps in the
low-learning-rate tail: adding the 15 k iterate to the average costs 0.018 BPB and adding the 10 k
one costs 0.042 BPB.

**Limits and negative results.** Capacity is the binding constraint on quality: width 288 / depth 8
and depth 10 variants train better but measured 4.7x the baseline scoring time, too close to the 5x
cap to submit, so the submitted model stops at 3.86x. Training saturates near 34 epochs — the last
iterate improves by only 0.01 BPB between 12.5 k and 15 k steps, and only averaging extracts more.
The three "free" architecture changes are not equally valuable: RMSNorm over LayerNorm is worth
0.001 BPB and SwiGLU over GELU 0.006 BPB, i.e. essentially nothing at this scale, whereas rotary
positions are worth 0.048 BPB; the report therefore claims the optimiser, the averaging and the
distillation as the mechanisms, not the block design.

**Fairness.** No test text was used at any point: hyperparameters, the decision to adopt
distillation, and the choice among {last iterate, EMA, SWA, soups} were all made on validation. The
test split was scored once, after freezing, and the same checkpoint reproduces identically on CPU.

## 9. Reproduction

```bash
cd code
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v

# 1. teacher (training-time only, never evaluated)
python train_student.py --config configs/teacher_m.json --steps 12000 --batch-size 32 \
  --device cuda --optimizer muon --lr 0.02 --warmup 300 --dropout 0.1 --eval-every 2000 \
  --ema-decay 0.999 --ema-start 0.6 --swa-start 0.75 --swa-every 200 --save-all \
  --run-dir runs/teacher

# 2. submitted student (distillation + Muon + EMA/SWA)
python train_student.py --config configs/student_d.json --steps 15000 --batch-size 32 \
  --device cuda --optimizer muon --lr 0.02 --embedding-lr 0.0003 --warmup 300 --dropout 0.1 \
  --eval-every 2500 --ema-decay 0.999 --ema-start 0.6 --swa-start 0.7 --swa-every 150 \
  --save-all --teacher-checkpoint runs/teacher/checkpoint.pt --alpha 0.5 --temperature 1.0 \
  --run-dir runs/final_run

# 3. freeze the selected weights (EMA) and score
python finalize.py runs/final_run/checkpoint_ema.pt --out runs/final --train-tokens 122880000
python evaluate.py --checkpoint runs/final/checkpoint.pt --device cpu --precision fp32 --split test
```

**AI assistance.** The architecture, optimiser, training recipe, experiment scripts and this report
were developed with the help of WorkBuddy, an AI coding assistant, used for implementation,
scripting and drafting. Every experiment was run and verified by the author using only the supplied
code, data and tokenizer.
