# Experiment evidence — every run behind REPORT.md

Generated from `evidence/<run>.metrics.json`, written automatically by each training script.
All figures in the table are **validation** BPB — the split used for development and for choosing
between the last iterate, EMA and SWA. The ranking metric, full-test BPB, appears exactly once,
in `submission/test_cpu_fp32.json`, and was computed only after the method was frozen.

| run | tag | model | params | steps x batch | optimiser | teacher | train targets | minutes | last | EMA | SWA | picked | validation BPB |
|---|---|---|---|---:|---|---|---:|---:|---:|---:|---:|---|---:|
| baseline_cpu | — | supplied baseline GPT (width 128, depth 4, learned positions) | 1,088,256 | 1200 x 32 | AdamW lr 2e-3 | — | 9,830,400 | 6.3 | n/a | n/a | n/a | — | **2.0711** |
| e1_adamw | adamw-lr002 | w256 d6 h4 rmsnorm/rope/swiglu | 5,247,744 | 6000 x 32 | adamw lr 0.002 | — | 49,152,000 | 7.4 | 1.5753 | n/a | n/a | final | **1.5753** |
| e1_muon | muon-lr02 | w256 d6 h4 rmsnorm/rope/swiglu | 5,247,744 | 6000 x 32 | muon lr 0.02 | — | 49,152,000 | 9.0 | 1.5240 | n/a | n/a | final | **1.5240** |
| e5_control | control-isotargets | w256 d6 h4 rmsnorm/rope/swiglu | 5,247,744 | 1200 x 32 | muon lr 0.02 | — | 9,830,400 | 1.8 | 1.7642 | n/a | n/a | final | **1.7642** |
| e5_muon_long | muon-long-20k | w256 d6 h4 rmsnorm/rope/swiglu | 5,247,744 | 20000 x 32 | muon lr 0.02 | — | 163,840,000 | 31.8 | 1.5105 | 1.4712 | 1.4708 | swa | **1.4708** |
| e5_teacher_m | teacher-m | w320 d8 h5 rmsnorm/rope/swiglu | 10,488,640 | 12000 x 32 | muon lr 0.02 | — | 98,304,000 | 36.7 | 1.4987 | 1.4551 | 1.4537 | swa | **1.4537** |
| e7_distill_d8 | distill-d8 | w256 d8 h4 rmsnorm/rope/swiglu | 6,822,144 | 15000 x 32 | muon lr 0.02 | 10,488,640 params, alpha 0.5 | 122,880,000 | 44.8 | 1.4597 | 1.4384 | 1.4441 | ema | **1.4384** |
| e7_gelumlp | ablation-gelumlp | w256 d6 h4 rmsnorm/rope/gelu | 5,246,208 | 4000 x 32 | muon lr 0.02 | — | 32,768,000 | 6.0 | 1.5547 | n/a | n/a | final | **1.5547** |
| e7_layernorm | ablation-layernorm | w256 d6 h4 layernorm/rope/swiglu | 5,251,072 | 4000 x 32 | muon lr 0.02 | — | 32,768,000 | 6.9 | 1.5499 | n/a | n/a | final | **1.5499** |
| e7_learnedpos | ablation-learnedpos | w256 d6 h4 rmsnorm/learned/swiglu | 5,313,280 | 4000 x 32 | muon lr 0.02 | — | 32,768,000 | 6.4 | 1.5966 | n/a | n/a | final | **1.5966** |
| e8_reference4k | reference-4k | w256 d6 h4 rmsnorm/rope/swiglu | 5,247,744 | 4000 x 32 | muon lr 0.02 | — | 32,768,000 | 6.5 | 1.5488 | n/a | n/a | final | **1.5488** |
| e9_nodistill_d8 | nodistill-d8-control | w256 d8 h4 rmsnorm/rope/swiglu | 6,822,144 | 15000 x 32 | muon lr 0.02 | — | 122,880,000 | 33.9 | 1.5036 | 1.4638 | 1.4649 | ema | **1.4638** |

How to read it. `e5_control` is the iso-target comparison against the supplied baseline (identical
9,830,400 training targets). `e1_muon` vs `e1_adamw` isolates the optimiser. `e7_layernorm`,
`e7_learnedpos`, `e7_gelumlp` each switch off one mechanism relative to `e8_reference4k` at the
same 4,000-step budget. `e9_nodistill_d8` vs `e7_distill_d8` isolates distillation at identical
budget. `e5_teacher_m` is the teacher, used only while training. Analysis in
[REPORT.md](../REPORT.md); the batch scripts that produced each row are the `exp*.sh` files.

## Score of the submitted checkpoint

Full test split, FP32, CPU, supplied scorer — the number reported to the leaderboard:

```json
{
  "bpb": 1.4569095766970173,
  "token_ppl": 21.022279495008053,
  "nll_nats": 1304742.901006421,
  "targets": 428405,
  "utf8_bytes": 1292013,
  "seconds": 32.93227769999794,
  "protocol": "7506-mp1-wt2-v2",
  "split": "test",
  "precision": "fp32",
  "checkpoint_sha256": "be575a224947bf58962f4fd5d60e0d6d6a3ea43cf34f8ca83ed3d8d2ad4c9373",
  "implementation_sha256": "29bd1cfb9a6c08a2451f5788abcaf82e90816172d65be9c98b386fd93d00744e",
  "evaluator_sha256": "128bcb2dab0be0d427505bddb4671e3ab3a8f78e114be79a689c0f9029af133d",
  "tokenizer_sha256": "020d1bc6aa4449c4f352b2e03d0e0fb4f39287f15297705e421b1fa7d817262e",
  "device": "cpu",
  "device_name": "CPU",
  "peak_allocated_gb": 0.0,
  "peak_reserved_gb": 0.0
}
```

Validation split, same checkpoint (development only — **not** a leaderboard score):

```json
{
  "bpb": 1.4384399743187737,
  "token_ppl": 20.891931733547715,
  "nll_nats": 1144621.0826234254,
  "targets": 376599,
  "utf8_bytes": 1148007,
  "seconds": 37.93631599999935,
  "protocol": "7506-mp1-wt2-v2",
  "split": "validation",
  "precision": "fp32",
  "checkpoint_sha256": "be575a224947bf58962f4fd5d60e0d6d6a3ea43cf34f8ca83ed3d8d2ad4c9373",
  "implementation_sha256": "29bd1cfb9a6c08a2451f5788abcaf82e90816172d65be9c98b386fd93d00744e",
  "evaluator_sha256": "128bcb2dab0be0d427505bddb4671e3ab3a8f78e114be79a689c0f9029af133d",
  "tokenizer_sha256": "020d1bc6aa4449c4f352b2e03d0e0fb4f39287f15297705e421b1fa7d817262e",
  "device": "cpu",
  "device_name": "CPU",
  "peak_allocated_gb": 0.0,
  "peak_reserved_gb": 0.0
}
```

## File map

| file | contents |
|---|---|
| `<run>.metrics.json` | hyperparameters, validation history, `checkpoint_sha256`, timings, device |
| `<run>.log` | raw per-step training and validation log |
| `baseline_cpu_train.log` | output of the supplied baseline trainer |
| `RESULTS.md` | this file |
