# Scratchpad History

CS886 (Fall 2026) course project: **how much scratchpad history do transformers need to learn
and generalize?**

We train a small transformer from scratch on cycle reachability (does following edges from node
`s` reach node `t`?) and vary only how much of its own scratchpad the model sees when it writes
the next state:

- **full history**: the question and every earlier state;
- **inductive (`k = 1`)**: the question and only the latest state.

Everything else (graphs, model, optimizer, step budget, test sets) is the same in both conditions.

- Research design: `PROJECT_SPEC.md`
- Execution plan, results and decision log: `plan.md`
- Invariants and conventions for contributors: `CLAUDE.md`

## Setup

Requires Python ≥ 3.10. With [uv](https://docs.astral.sh/uv/):

```bash
uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```

With plain `pip`, create and activate a Python ≥ 3.10 virtual environment, then run
`pip install -e ".[dev]"`.

Training uses a CUDA GPU if there is one, then Apple `mps`, then the CPU.

## Check the install

```bash
pytest                            # 58 tests, a few seconds
python scripts/smoke_overfit.py   # memorizes 100 problems for both conditions (~3 min)
```

## Run one training run

```bash
python -m scratchpad.train --k 1 --n-train 16000 --seed 0   # inductive
python -m scratchpad.train --k 0 --n-train 16000 --seed 0   # full history (--k 0)
```

| Flag | Meaning | Default |
|---|---|---|
| `--k` | history length: `1` = inductive, `0` = full history | `0` |
| `--n-train` | number of training problems | `16000` |
| `--seed` | seed for the training graphs and the model | `0` |
| `--max-steps` | optimizer step budget | `4000` |
| `--device` | `auto`, `cpu`, `mps` or `cuda` | `auto` |

A run takes about 8–15 minutes on an Apple-silicon laptop. It writes to `results/<run_id>/`,
where `run_id` is for example `k1_n16000_s0` or `kfull_n16000_s0`:

- `config.json`: the full configuration;
- `metrics.json`: the training log and the held-out accuracy at 3, 4, 5 and 6 steps
  (written last, so its presence marks the run as finished);
- `model.pt`: the trained weights.

All other settings (task size, model size, optimizer) are dataclass defaults in
`src/scratchpad/config.py`.

## Reproduce the experiments

```bash
python scripts/run_sweep.py --dry-run     # print the 24-run matrix without training
python scripts/run_sweep.py               # run all of it (about 5 hours on a laptop)
```

The matrix is 2 conditions × 4 training sizes (250, 1,000, 4,000, 16,000) × 3 seeds. The sweep
skips any run whose `metrics.json` already exists, so it can be stopped and restarted. To run a
part of it:

```bash
python scripts/run_sweep.py --ks 1,full --sizes 16000 --seeds 0,1,2   # the six main runs (~50 min)
```

## Attention analysis

For a finished run, measure how much attention goes to the one edge that matters at each step:

```bash
python -m scratchpad.attention --run results/k1_n16000_s0
```

This writes `attention.json` into the run folder. Run it for each run you want in the attention
figure (the figure uses the runs at `--n-train`, 16,000 by default).

## Tables and figures

```bash
python scripts/make_plots.py
```

This reads every finished run in `results/` and writes:

- `results/summary.csv`: one row per run and test length;
- `results/figures/length_generalization.png`: accuracy by problem length;
- `results/figures/sample_efficiency_train.png` and `sample_efficiency_longest.png`: accuracy by
  number of training problems;
- `results/figures/attention_share.png`: attention on the correct edge, for runs that have an
  `attention.json`.

Folders in `results/` whose names start with `_` (pilot runs) are ignored.

## Layout

```
src/scratchpad/
  config.py     defaults for the task, model and training
  task.py       graph generator and oracle trace
  tokens.py     vocabulary, encode/decode
  formats.py    problem → training examples for a given history length
  data.py       datasets, padding, held-out sets
  model.py      the small Llama-style model
  train.py      training loop and CLI
  evaluate.py   step-by-step solving and scoring
  attention.py  edge-attention share and CLI
scripts/        smoke_overfit.py, run_sweep.py, make_plots.py
tests/          pytest suite
results/        run outputs (not committed)
papers/         source papers (not committed)
```

## Source papers

- E. Abbe et al., "How Far Can Transformers Reason? The Globality Barrier and Inductive
  Scratchpad", NeurIPS 2024, arXiv:2406.06467.
- K. Wen et al., "From Sparse Dependence to Sparse Attention: Unveiling How Chain-of-Thought
  Enhances Transformer Sample Efficiency", ICLR 2025, arXiv:2410.05459.
