# CLAUDE.md

CS886 (Fall 2026) course project: **How Much Scratchpad History Do Transformers Need to
Learn and Generalize?** We train small transformers from scratch on cycle-graph
reachability and vary how many earlier scratchpad states the model can see.

- Research design: `PROJECT_SPEC.md`
- Execution plan, experiment matrix, and decision log: `plan.md` (keep its checkboxes and §9 current)
- Source papers: `papers/` (arXiv 2410.05459 "From Sparse Dependence to Sparse Attention";
  arXiv 2406.06467 "How Far Can Transformers Reason?")

## Setup and commands

Requires Python ≥ 3.10. The system `python3` here is 3.9, so use a venv:

```bash
uv venv --python 3.11 .venv && source .venv/bin/activate   # no python3.11 on PATH; uv fetches it
uv pip install -e ".[dev]"

pytest                                   # all tests
pytest tests/test_task.py -q             # one file
python scripts/smoke_overfit.py          # milestone 2 exit check: memorize 100 instances (~3 min)
python -m scratchpad.attention --run results/k1_n16000_s0     # edge-attention share; writes attention.json
python -m scratchpad.train --k 1 --n-train 16000 --seed 0     # one run (k=0 means full history; up to ~30 min)
python scripts/run_sweep.py --dry-run    # print the experiment matrix
python scripts/run_sweep.py --ks 1,full --sizes 16000 --seeds 0,1,2   # milestone 3 (hours; resumable)
python scripts/make_plots.py             # regenerate figures from results/
python scripts/threshold_table.py --results-dir results   # steps/examples/tokens to 95% validation accuracy
```

## Layout

```
src/scratchpad/
  config.py     dataclass configs (task, model, train); single source of defaults
  task.py       graph generator + oracle trace (pure Python, no torch)
  tokens.py     vocabulary, encode/decode, state parsing (pure Python)
  formats.py    instance → training examples for a given history k, with edge positions
  data.py       torch Dataset, padding collate, held-out set construction
  model.py      tiny LlamaForCausalLM builder
  train.py      training loop + CLI; writes results/<run_id>/
  evaluate.py   step-by-step solving loop and scoring
  attention.py  edge-attention share
  utils.py      seeding, device selection, run IDs, JSON I/O
scripts/        smoke_overfit.py, run_sweep.py, make_plots.py
tests/          pytest; task/tokens/formats tests define milestone 1, data/evaluate/train tests milestone 2
results/        run outputs (git-ignored)
papers/         PDFs (git-ignored)
```

## Invariants (do not break these)

1. **The state is Markov.** A state is `S <node> <status>`. Question + latest state must be
   enough to compute the next state. If the format changes, this must still hold.
2. **Only history varies between conditions.** Same graphs, state format, model, optimizer,
   step budget, and eval sets for every k. Anything else that differs is a confound.
3. **History is restricted in the data** (`formats.make_examples`), never by changing the model.
   `k = None` (CLI: `--k 0`) means full history; `k = 1` is the inductive scratchpad.
4. **The oracle never helps at test time.** `evaluate.solve` feeds the model its *own*
   previous states. The oracle is used only for data generation and scoring.
5. **Loss is only on target-state tokens.** Context labels are `-100`.
6. **Held-out sets are disjoint from training** (dedupe by graph + query hash) and fixed
   across runs (fixed seed).
7. **Edges are shuffled** in every instance so edge order never reveals the path.
8. **Attention success ≠ low entropy.** Use edge-attention share normalized over edge tokens
   (`plan.md` §4).
9. `task.py`, `tokens.py`, `formats.py` stay **torch-free** so they're fast to test.

## Conventions

- Type hints everywhere; `from __future__ import annotations` at the top of each module.
- Configs are dataclasses in `config.py`; don't hard-code hyperparameters in other modules.
- Every randomness source takes an explicit `random.Random` / seed; no global RNG use.
- Each run writes `results/<run_id>/config.json` and `metrics.json`; `run_id` encodes k, N, seed.
- Small, testable functions; add a test with each new behaviour.
- Don't commit results, checkpoints, or PDFs.

## Working with the user

- The user is an experienced software engineer who is still learning transformers. Define
  unfamiliar ML terms the first time they come up. When writing math, also say how to read it aloud.
- Explain decisions and flag confounds. Prefer small, first implementations.
- **Ask before:** choosing or enlarging model size, starting long or paid compute (the full
  sweep, cloud GPUs), or adding tasks or conditions beyond `plan.md`.
- The task construction was checked against the second paper on 2026-11-17 (`plan.md` §2, §9).
  The first paper has not been checked yet. Known deviations from the second paper are logged in §9.

## Status

- 2026-10-20: plan written, scaffold created.
- 2026-11-17: milestones 1 and 2 done (`plan.md` §10). Graph construction switched to the second
  paper's. `pytest` passes (48); the model (4.2M parameters) memorizes 100 instances through
  `evaluate.solve` for k=1 and full history. Still stubs: `attention.py`,
  `scripts/make_plots.py`.
- 2026-11-17: milestone 3 started, no results yet. The user approved the 4.2M model and ~3 h of
  laptop compute. `scripts/run_sweep.py` now runs (filters, resumable). Short k=1 pilots were
  stopped by the user while the loss was still flat (~1.24, 0% accuracy by step 1,000), so it is
  unknown whether the default settings learn within 20,000 steps; see the open issue in `plan.md`
  §5 week 2. The user will rerun later. Changing batch size or learning rate needs their approval.
- 2026-11-20: the user approved shrinking the task to 16 nodes (train 4 steps, test 4–8) and the
  paper-like training defaults (batch 512, cosine lr, weight decay 0.1); see `plan.md` §9. History
  conditions are now k ∈ {1, 2, full}. Step counts in the commands and text above that say 6 or 12
  predate this. Two k=1 pilots in `results/_pilot_16nodes/` did not learn.
- 2026-11-21: the user approved dropping to 8 nodes (train 3 steps, test 3–6); conditions are now
  k ∈ {1, full}, budget 4,000 steps (`plan.md` §9). The six gate runs were started.
- 2026-11-22: the gate passed (`plan.md` §5 week 2). At 8 nodes, N=16k, 3 seeds: k=1 solves 3–6 steps
  at ~100%; full history is 100% at 3 steps and 0% trace accuracy at 4–6 (it ends every walk at the
  third state). `attention.py` and `scripts/make_plots.py` are implemented; `pytest` passes (58).
  `plan.md` §1–§3 now describe the 8-node version. Not started: the main sweep (its training sizes
  need the user's decision, see `plan.md` §3) and the slides.
- 2026-11-22: main sweep done (24 runs, N ∈ {250, 1k, 4k, 16k}; `plan.md` §5 week 3). Neither condition
  learns at N ≤ 1k; at N=4k full history learns in 3 of 3 seeds and k=1 in 1 of 3; both learn at 16k.
  Full history never generalizes past 3 steps. Slides drafted as a claude.ai artifact
  (https://claude.ai/artifact/RtgwAL5CgJbN6HLbHQyjHP). The first paper is still unchecked.
- 2026-11-22: `README.md` rewritten as the how-to-run guide. The 3-page report is a claude.ai doc
  (https://claude.ai/artifact/HaUD8CMbKSPphy3iTcTuh9) and is deliberately not in the repo. The deck now
  has 12 slides for a 15-minute talk.
- 2026-11-24: the user has about two more weeks and wants to close the gaps against the submitted
  proposal (24 nodes, 6 → 12 steps, history ∈ {full, 4, 2, 1}, a separate position-reset test,
  examples/tokens to a validation threshold). `train.py` gained `--fresh` (new training instances every
  step; pilots only), `--n-nodes`, `--train-steps`, `--test-steps`, `--eval-every`, `--results-dir`;
  `pytest` passes (62). A 16-node / 6-step / k=1 fresh-data pilot (≤ 50,000 steps, ≤ ~8 h) was started
  into `results/_pilot_16nodes_6step_fresh/` (progress in its `train.log`). Next steps depend on it and
  each needs the user's approval (`plan.md` §9). A study-notes deck for the user is at
  https://claude.ai/artifact/DGghaKYfk814AhWCdJQQH7; the paper PDF is in `~/Downloads`.
- 2026-11-24: that pilot learned: plateau until step ~8,500, then ≥99.7% trace accuracy at 6, 8, 10
  and 12 steps; stopped early at 15,000 steps (117 min). One seed, k=1 only, fresh data. Not yet
  approved or run: the other history conditions (full, 4, 2) at this size, more seeds, 24 nodes.
- 2026-11-24 (evening): the user approved the full, 4 and 2 conditions at that size (seed 0, same
  settings); started as one sequential job, logs in `results/_pilot_16nodes_6step_fresh/train_k{0,4,2}.log`.
  More seeds and 24 nodes are still unapproved.
- 2026-11-25: those runs finished (seed 0). Full history: 100% at 6 steps, 0% trace accuracy at 8–12
  (always fails at the sixth state). k=4 and k=2: 100% at 6–12 and reached 95% validation accuracy by
  step 2,000–2,500, like full history; k=1 needed 10,000. Numbers in `plan.md` §9. One seed only.
- 2026-11-25: the user approved seeds 1 and 2 for the four conditions at 16 nodes; the eight runs were
  started (logs `train_k<k>_s<seed>.log` in the same folder). Added `scripts/threshold_table.py`;
  `pytest` passes (63). Seed-0 attention: best-head share ≈ 0.99 for full, 4 and 2, but 0.39 for k=1
  (`plan.md` §9). Not approved: position-reset test, 24 nodes. Paper and slides not yet updated.
- 2026-11-25: the user may have access to the Alliance (Compute Canada) Fir cluster and wants to train
  there. `scripts/cluster/setup_env.sh` and `scripts/cluster/train_array.sh` (Slurm array: 4 conditions ×
  3 seeds, one GPU each) were written but are untested: the Alliance docs could not be fetched, so module
  names, GPU request syntax and wheel versions are unverified. Needed from the user: confirmed login,
  username and the `def-...` account name. Only the first of the eight laptop seed runs had finished by 10:50.
- 2026-11-25: Fir works (user `fzaqout`, account `def-ssalihog_gpu`, code in `~/scratch/cs886`, reached
  through an ssh control socket the user opens with Duo). The cluster scripts now build the virtualenv in
  `$SLURM_TMPDIR` inside each job (the user asked for fast local storage, not home or scratch) and copy
  results back at the end. `scripts/cluster/smoke_test.sh` passed (63 tests, cuda, 10 GB H100 slice;
  step speed roughly the laptop's). The 12-run array (`train_array.sh`) was submitted on the user's approval as job 6093788
  (4 conditions × seeds 0–2, results in `results/_cluster_16nodes_6step_fresh/` on Fir); the laptop queue
  was stopped (`kfull_n16000_s1` there is incomplete). Results still have to be copied back and reported.
- 2026-11-25: the user decided that all reported results must come from Fir. The 8-node sweep was
  resubmitted there (array 6123940 → `results/_cluster_8nodes_3step/` on Fir, attention included); 11 of
  the 12 16-node runs are done and copied to `results/_cluster_16nodes_6step_fresh/` (k=1 seed 2 still
  running, stalled near 30%). Laptop results in `results/` are now a cross-check only. To do: copy results
  back, run `scripts/cluster/attention_job.sh` for the 16-node runs, then rewrite paper and slides.
- 2026-11-25: all 36 Fir runs are finished and copied back with attention (`results/_cluster_16nodes_6step_fresh/`,
  `results/_cluster_8nodes_3step/`); numbers in `plan.md` §9. Headlines: full history never generalizes past the
  training length (both sizes, all seeds); k=4, 2 and 1 all generalize to 12 steps; steps to 95% validation are
  1,500 for k=4 in all seeds and up to 25,500 for k=1; at 8 nodes N=4,000, k=1 learns in 2 of 3 seeds on Fir.
  The "inductive attends more sharply" claim is not supported on Fir. Next: rewrite the paper and slides from the
  Fir numbers (`make_plots.py` titles assume fixed N and need adjusting for the fresh-data runs). Cluster jobs now
  request 6 GB of RAM. Not approved: position-reset test, 24 nodes, more seeds.
- 2026-11-25: the paper was rewritten from the Fir results as `~/Downloads/CS886_scratchpad_history_report_v2.pdf`
  (+ `.tex`, 3 pages; the 22 Nov version is kept beside it): Experiment A = 16 nodes / four history lengths /
  fresh data, Experiment B = the 8-node fixed-data sweep, attention as a table. `make_plots.py` now handles four
  conditions and fresh-data runs. Still describing the old laptop results and needing an update: the
  presentation deck, the PowerPoint, the study-notes deck, the online report and `README.md`.
- 2026-11-25: the presentation was rewritten from the Fir results: the online deck
  (https://claude.ai/artifact/RtgwAL5CgJbN6HLbHQyjHP) now has 13 slides (new `cost` slide; attention is a
  table) and the matching PowerPoint is `results/slides/scratchpad_history_v2.pptx` (the 22 Nov file is kept).
  Still describing the old laptop results: the study-notes deck, the online report artifact and `README.md`.
- 2026-11-25 (late afternoon): the user approved rerunning Experiment A at the proposal's 24 nodes on Fir
  (`sbatch --time=12:00:00 --job-name=scratchpad-24n --export=ALL,N_NODES=24 scripts/cluster/train_array.sh`).
  The ssh control connection died during the submit, so it is unknown whether the array was queued: check
  `squeue`/`sacct` before submitting again. The user must reopen the connection (Duo). The report was also
  restyled in the ICML 2025 template, at the user's request to match a classmate's layout:
  `~/Downloads/CS886_report_icml/` (tex, pdf, style files; 3 pages; no email or affiliation line).
- 2026-11-25: after the user reconnected, the queue was checked (nothing had been submitted) and the 24-node
  array was submitted once as job 6173708 → `results/_cluster_24nodes_6step_fresh/` on Fir. When it finishes:
  copy back, run `attention_job.sh` on that folder, and update the ICML report and the slides.
- 2026-11-25 (evening): a whole H100 is ~4.5× faster than the 10 GB slice but waited 50 min in the queue.
  `scripts/cluster/train_array.sh` now defaults to `--gres=gpu:h100:1` and a 3-hour limit (override on the
  sbatch line for a slice). The 24-node array 6173708 keeps running on slices: 3 of 12 done by 19:20 EST,
  the 4.2M model does learn at 24 nodes; several runs (all k=1) were still on the plateau at ~18,000 steps.
- 2026-11-25 20:10 EST: per the user, the 8 unlearned 24-node runs were also queued on whole H100s (array
  6210841 → `results/_cluster_24nodes_6step_fresh_h100/`). `scripts/cluster/swap_watch.sh` runs on Fir for
  one hour (log `slurm_logs/swap_watch_6210841.log`): it cancels a slice task only when its H100 twin has
  started, and cancels H100 tasks still pending after the hour. Check that log before touching either array.
- 2026-11-25 21:10 EST: the hand-over hour ended with no whole-H100 task started; the watcher cancelled all 8
  tasks of array 6210841 and no slice was stopped. Slice array 6173708 carries on (8 running, ~29,000–32,000 of
  50,000 steps, 12-hour limit): k=4 seed 2 has learned; k=2 seeds 1 and 2 and k=1 seed 1 have started to move;
  full seed 0, k=2 seed 0 and k=1 seeds 0 and 2 are still flat. No results folder `_h100` exists.
- 2026-11-26: the 24-node array 6173708 finished on slices and is copied to `results/_cluster_24nodes_6step_fresh/`
  (numbers in `plan.md` §9). 8 of 12 runs learned within 50,000 steps: full 2 of 3, k=4 3 of 3, k=2 1 of 3, k=1 2 of 3.
  Learned full-history runs score 0% past 6 steps; every learned limited-history run scores ≥ 95% at 12 steps.
  Unlike at 16 nodes, k=4 was not the fastest (full history was, in two seeds). Attention job 6256206 was submitted
  for that folder. The ICML report and the slides are being updated with these results.
- 2026-11-26: 24-node attention (job 6256206) is copied back. Best-head share at 12 steps, seeds 0/1/2, unlearned
  runs in parentheses: full (0.16)/0.96/0.04; k=4 0.72/0.99/0.11; k=2 (0.04)/(0.38)/0.04; k=1 (0.09)/0.80/0.95;
  chance 0.04. Three learned 24-node runs have no edge-attending head at the measured position. The ICML report
  (`~/Downloads/CS886_report_icml/`, now 4 pages with Experiment C; 16-node backup beside it) and the PowerPoint
  (`results/slides/scratchpad_history_v3.pptx`, 14 slides) include the 24-node results. The online deck was NOT
  republished (the publish was denied by a permission check); its edited source is in the session scratchpad, so
  it may need redoing: new `nodes24` slide after `cost`, and edits to setup, cost, attention, limits, takeaways.
- 2026-11-26: the user decided the final report should not mention the 16-node experiment. The ICML report now
  has Experiment A = 24 nodes and Experiment B = the 8-node fixed-data sweep (3 pages, new figure
  `training_curves_24.png`; earlier versions kept as `..._16and24_backup` and `..._16node_backup`). The slides
  (PowerPoint v3 and the online deck) still present the 16-node results; the user has not said to change them.
- 2026-11-26: the user decided not to run the position-reset test. The report's limitation now says that the
  windowed contexts are rebuilt at every step, so hiding history and resetting positions happen together and
  were not separated.
- 2026-11-26: the user wants every experiment rerun with a ~10M-parameter model matching Abbe et al. (6 layers,
  6 heads, width 384) on whole H100 cards, but said NOT to submit until they finish reviewing the paper and may
  add more changes. Prepared only: `train.py` has `--n-layers`, `--n-heads`, `--hidden-size`, `--ffn-size`; the
  cluster scripts take `MODEL_ARGS`. `--n-layers 6 --n-heads 6 --hidden-size 384` (feed-forward width left at
  1024) gives 10,647,936 parameters; `pytest` passes (64). Nothing synced to Fir, nothing submitted. Still
  Llama/RoPE, not the paper's GPT-2 with learned absolute positions.
