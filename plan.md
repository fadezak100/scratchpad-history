# Project Plan: Scratchpad History, Learning, and Length Generalization

CS886, Fall 2026. Timeline: 5 weeks, Tue 2026-10-20 to presentation (week of 2026-11-23; exact date TBD).
Research design and motivation: see `PROJECT_SPEC.md`. This file is the execution plan.

---

## 1. Goal in one paragraph

Train small transformers from scratch on cycle reachability (does following edges from
node `s` reach node `t`?). Training problems need 3 steps; test problems need up to 6
(8 nodes since 2026-11-21; originally 24 nodes, 6 → 12, then 16 nodes, 4 → 8, see §9).
Vary only how much earlier scratchpad history the model sees when writing the next state:
full history or only the latest state (inductive). With 3-step training there is no
intermediate "last k states" condition left (§9, 2026-11-21). For each condition,
measure (1) how many training examples it needs, (2) accuracy on longer problems, and
(3) how much attention goes to the one edge that matters at each step.

## 2. Fixed design decisions

These are decided. Change them only on purpose and record the change in §9.

| Decision | Choice | Reason |
|---|---|---|
| Graph | 8 nodes. For an L-step instance: NO = two cycles of length L and 8 − L (`s` in the L-cycle, `t` in the other); YES = one 8-cycle with `t` exactly L ahead of `s` | The second paper's construction (§3.2.1) scaled down from 24 nodes, because 24 and 16 did not train in the time available (§9, 2026-11-20 and 2026-11-21). YES and NO differ only globally, so the model has to walk |
| Query | `Q s t` after the shuffled edge list | Shuffling stops edge order from leaking the answer |
| State | `S <node> <status>`, status ∈ {GO, YES, NO} | Markov: state + question are enough for the next step |
| Answer | Status of the final state (YES = reached `t`, NO = returned to `s`) | No separate answer token needed |
| Train length | Exactly 3 steps, 50/50 YES/NO | Close to the second paper's 6 : 12 : 24 train : test : nodes ratio (ours is 3 : 6 : 8; 6 is the longest valid length, since L ≤ n − 2) |
| Test lengths | 3 (in-distribution), 4, 5, 6 | Gives a curve, not one point |
| History conditions | k ∈ {1 (inductive), full} | With 3-step training, k ≥ 2 already equals full during training |
| History mechanism | In the data: each example is "question + last k states → next state" | Positions reset automatically; no custom attention masks |
| Model | Tiny Llama (RoPE), 4.2M params: 4 layers, 4 heads, width 256 | RoPE avoids untrained absolute positions on long full-history inputs |
| Loss | Only on target-state tokens (context labels = -100) | Model is graded on writing the next state, not on copying context |
| Decoding | Greedy, one state at a time, run in a loop at test time | Model must use its own earlier states, not the oracle's |
| Seeds | 3 per cell (5 if time allows) | Report mean ± spread |
| Oracle role | Generates data and scores outputs only; never corrects the model | Keeps the study about learning, not about external repair |

## 3. Experiment matrix

**Main sweep (sample efficiency + length generalization):**

```
conditions     k ∈ {1, full}                     2
train sizes    N ∈ {250, 1k, 4k, 16k} instances    4
seeds          {0, 1, 2}                          3
                                                ----
                                                24 runs
```

Only about 80k distinct 3-step instances exist at 8 nodes (40,320 YES + 40,320 NO), so N = 64k
would cover most of the space and no longer test generalization to unseen graphs. It was replaced
by N = 250 at the low end (§9, 2026-11-22). The six N = 16k runs are the gate runs.

One "instance" is one graph + query and produces 3 training examples (one per step).
Every run is evaluated on fixed held-out sets: 2,000 instances each at 3, 4, 5 and 6 steps.

**Training budget rule:** every run gets the same max number of optimizer steps (4,000) and is
stopped early when 3-step validation accuracy plateaus at or above 95% (see §9, 2026-11-17). Log examples seen, tokens seen,
and an estimated FLOP count. This allows comparisons both per example and per token.

**Sample-efficiency metric:** the smallest N at which held-out 3-step final-answer
accuracy is ≥ 95%. Also plot the full curve of accuracy vs. N.

**Attention analysis:** on the best seed per condition at N = 16k (or the smallest N that
reaches 95%), across all test lengths.

## 4. Metrics (precise definitions)

- **Final-answer accuracy:** fraction of instances whose last generated status matches the oracle.
- **Trace accuracy:** fraction of instances where every generated state matches the oracle.
- **First-error step:** for wrong traces, the index of the first wrong state. Histogram per test length.
- **Format failures:** generated text that doesn't parse as a state. Counted as wrong and reported separately.
- **Edge-attention share:** at the position where the model predicts the next node, the
  attention on the correct edge's tokens divided by the attention on all edge tokens.
  Restricting to edge tokens means a shorter history doesn't inflate the score mechanically.
  Chance level is 1 / (number of edges) = 1/8. Report per layer and head; highlight the best head.
- **Not used as a success signal:** attention entropy on its own (masking history lowers it automatically).

## 5. Week-by-week plan

### Week 1 (Oct 20–26): Foundations
- [x] Put both PDFs in `papers/`. Check task construction, step counts, state format and model setup against the second paper. Update §2 if anything differs. (Second paper checked 2026-11-17; first paper not yet.)
- [x] Set up the environment (Python ≥ 3.10 venv, `pip install -e ".[dev]"`).
- [x] `task.py`: `make_instance`, `oracle_trace`, with tests (exact step count, YES/NO balance, valid graph).
- [x] `tokens.py`: vocabulary, encode/decode, state parsing, with round-trip tests.
- [x] `formats.py`: build examples for each k, with edge-position metadata; tests that context size matches k.
- [x] `data.py`: Dataset + padding collate; dedupe held-out sets against training graphs.
- [x] `model.py` + `train.py` minimal loop. **Exit check:** the model overfits 100 instances to ~100%.
- [x] Time one realistic run (k=1, N=16k). Use it to estimate the full sweep's compute. (Extrapolated from 200 steps; see §10.)

### Week 2 (Oct 27–Nov 2): Replication gate
- [x] `evaluate.py`: step-by-step solving loop, scoring, first-error step.
- [x] Train full vs. inductive at N = 16k, 3 seeds; evaluate at 3/4/5/6 steps (8 nodes; done 2026-11-22, results below).
  Command: `python scripts/run_sweep.py --ks 1,full --sizes 16000 --seeds 0,1,2` (resumable).
  **Open issue (2026-11-17):** no run has finished. Pilots at k=1 were stopped by the user while the
  loss was still flat at ~1.24 (format learned, next node guessed at random, 0% accuracy): batch 64 /
  lr 3e-4 at step 1,000, and batch 256 / lr 1e-3 at step 375. It is not yet known whether the default
  settings learn the edge lookup within 20,000 steps. Check that on one run before launching all six.
  **Pilot findings (2026-11-17, all k=1, Apple `mps`, ~800 examples/s at any batch size):**
  - Default settings (batch 64, lr 3e-4 constant), 24 nodes, N=16k, full 20,000 steps: loss 1.24 → 1.09,
    0% trace accuracy at every length. Output kept in `results/_failed_default_settings/`.
  - Paper-like settings (batch 512, cosine 3e-4 → 3e-5, clip 1.0, weight decay 0.1, beta2 0.99), 24 nodes,
    1,400 steps: loss 1.05, final accuracy ~50%, trace 0%. Identical for N=16k and for ~fresh data
    (N=120k), so data volume is not the limit there. Loss 1.05 = format and status learned, next node
    guessed among 23.
  - Same but lr 1e-3: worse (loss 1.13 at step 1,400).
  - Paper-like settings, **8 nodes / 3-step training**, fresh data: flat at 0.62 until step ~700, then
    100% trace accuracy on unseen graphs by step 900. So the pipeline can learn the lookup; at 24 nodes
    the plateau is simply longer than we have trained.
  - The paper's code (github.com/aryol/inductive-scratchpad, `config/cycles_ood.py`): batch 512 whole
    instances, fresh samples, 20,000 iterations, dropout 0.2, 6 layers / 6 heads / width 384.
  **Decided 2026-11-20:** shrink to 16 nodes / 4-step training / test 4–8, and make the paper-like
  optimizer settings the defaults (§9). A k=1 pilot (N=16k, capped at 5,000 steps) writes to
  `results/_pilot_16nodes/`.
  **Pilot result (2026-11-20, k=1, N=16k, 5,000 steps, 2,290 s):** training loss fell 1.01 → 0.18 but
  held-out trace accuracy stayed 0% at every length and final accuracy fell from 52% to ~33%; 88% of
  traces are wrong at the first step. 5,000 steps × 512 is 40 passes over 64k examples, so the model
  memorized the training graphs instead of learning the lookup. A second pilot with N=160k (4 passes)
  was run to check whether 16 nodes learns at all with enough data.
  **Pilot 2 result (2026-11-20, k=1, N=160k, 5,000 steps, 2,225 s):** no memorization, but no learning
  either. Loss flat at 0.90 → 0.86 from step 1,000 on, trace accuracy 0%, final accuracy ~50%, 87% of
  traces wrong at the first step. So at 16 nodes the lookup plateau is longer than 5,000 steps (the
  cosine schedule had also decayed the learning rate to 3e-5 by then). Only 8 nodes / 3-step training
  is confirmed to learn. **Decided 2026-11-21:** drop to 8 nodes (§9). The gate
  (`run_sweep.py --ks 1,full --sizes 16000 --seeds 0,1,2`) was started that day. The 16-node 20,000-step
  run was not started.
  Still open: `max_steps` is 20,000, which at batch 512 is far longer per
  run than the old estimate in §10; pick the sweep budget from the pilot.
  **Gate result (2026-11-22, 8 nodes, N=16k, 3 seeds each, all six runs stopped early at 2,000–2,250 steps,
  ~8–9 min each on `mps`):**

  | Condition | 3 steps (trained) | 4 steps | 5 steps | 6 steps |
  |---|---|---|---|---|
  | k=1, trace accuracy | 100% | 100% (one seed 99.95%) | 100% | 100% |
  | full, trace accuracy | 100% | 0% | 0% | 0% |
  | full, final-answer accuracy | 100% | ~50% | ~50% | ~50% |

  Every full-history failure is at the third state (first-error step index 2, all 2,000 instances, all seeds).
  On 6-step problems the full-history models write the first two states and the third node correctly, then
  mark the third state YES or NO instead of GO. So they follow edges correctly but end the walk by counting
  states, not by comparing the node with `s` and `t` (the shortcut predicted in §7). No format failures.
  Edge-attention share of the best head (chance 1/8): k=1 is 0.99 at every step and length in all seeds;
  full is 0.88–0.97 on steps 1–4 and drops to 0.72–0.91 on steps 5–6. Caveat: the attention contexts hold
  the oracle's states, and the full-history models never get past step 3 on their own.
- [x] **Decision gate (end of week 2):** inductive clearly beats full at the longest length (6 steps)? **Yes** (2026-11-22).
  - **Yes** → proceed to week 3.
  - **No** → spend at most 3 days debugging (see §7), then present whatever pattern appears as the finding.

### Week 3 (Nov 3–9): Main sweep
- [x] `scripts/run_sweep.py`: run the 24-run matrix (finished 2026-11-22, results below), resumable, one `results/<run_id>/` per run.
- [x] Collect metrics into one table (`results/summary.csv`, written by `scripts/make_plots.py`).
- [x] First plots: accuracy vs. N, accuracy vs. test length per condition (`results/figures/`).

  **Sweep result (2026-11-22, 24 runs).** Seeds (of 3) that solve held-out 3-step problems; every run
  scored either 100% or under 3% trace accuracy:

  | N | k=1 | full |
  |---|---|---|
  | 250 | 0 | 0 |
  | 1k | 0 | 0 |
  | 4k | 1 | 3 |
  | 16k | 3 | 3 |

  Failed runs memorized (training loss reached 0, held-out trace accuracy near 0), so the two k=1 failures
  at N=4k were not cut off by the 4,000-step budget. So at the training length full history needed no more
  data than k=1, and at N=4k it learned more reliably (3 seeds only, so this is suggestive). Full history
  scored 0% trace accuracy at 4–6 steps at every N; the one k=1 seed that learned at N=4k scored 100% at
  3–6 steps. Attention was run on the N=16k runs only.

### Week 4 (Nov 10–16): Attention + analysis
- [x] `attention.py`: edge-attention share per layer/head (`python -m scratchpad.attention --run results/<run_id>` writes `attention.json`).
- [ ] Attention plots: share vs. step index, for each condition and test length.
- [ ] Error analysis: first-error histograms. Does full history fail at a characteristic step (e.g. step 7)?
- [ ] Freeze the main figures. Extra seeds only if compute allows.

### Week 5 (Nov 17–23): Write-up + presentation
- [ ] Slides: question → setup → replication → sweep → attention → limitations.
- [ ] Short report (if required).
- [ ] No new experiments after Wed Nov 19, except re-running a broken cell.

## 6. Deliverables

1. Code in this repo; `pytest` passes.
2. `results/summary.csv` + figure scripts that regenerate every plot.
3. Three core figures: sample efficiency (accuracy vs. N), length generalization (accuracy vs. steps), attention share.
4. Slides and, if required, a short report.

## 7. Risks and fallbacks

| Risk | Early sign | Fallback |
|---|---|---|
| No condition generalizes to 12 steps | Inductive fails at 8 in week 2 | Check state format, eval loop, decoding; try mixed training lengths 1–6; report a curve over 6–12 instead of only 12 |
| Full history also generalizes | Full ≈ inductive in week 2 | Still a result; focus on sample efficiency + attention differences |
| Sweep too slow | Week-1 timing shows > 30 min per run | Drop N = 64k, or 2 seeds, or a smaller model |
| Model learns a shortcut (e.g. "always stop at 6") | Full history: high 6-step accuracy, collapse at step 7 | This is an expected finding; show it with first-error histograms |
| Attention scores are noisy or spread out | No single head stands out | Report the max over heads plus averages; check the per-layer pattern |
| Paper details differ from §2 | Week 1 check | Follow the paper; record the change in §9 |

## 8. Open questions (need answers)

- Exact presentation date and whether a written report is required.
- Team size and who owns what (suggested split: one person on data/eval, one on training/sweep, both on analysis).
- Compute: laptop (Apple `mps`) vs. Colab vs. a university GPU. Decide after the week-1 timing run.
- Does the professor want any baseline beyond the second paper's comparison?

## 9. Decision log

| Date | Decision | Why |
|---|---|---|
| 2026-10-20 | Dropped checker/repair design; train-from-scratch history sweep | Align with the two source papers' questions |
| 2026-10-20 | History restricted in data, not attention masks | Simpler; position reset comes for free |
| 2026-10-20 | Edge-attention share normalized over edge tokens | Avoid the entropy confound from masking |
| 2026-11-17 | Graph construction switched to the second paper's: NO = cycles of L and 24 − L, YES = one 24-cycle with `t` L ahead | §7 rule "follow the paper"; replaces random disjoint cycles |
| 2026-11-17 | Kept deviation: state is `S <node> <status>` (paper: current node only, then `;0`/`;1`) | Same information; easier to parse and score |
| 2026-11-17 | Kept deviation: 24 fixed node labels (paper: 24 names drawn from a pool of 1000) | Smallest first implementation; revisit if results look label-specific |
| 2026-11-17 | Kept deviation: Llama + RoPE, 4 layers / 4 heads / width 256, batch 64 (paper: GPT-2 style, 6 / 6 / 384, ~10M, learned absolute positions, large batches) | Already decided in §2; raise batch size first if training is unstable |
| 2026-11-17 | Early stopping uses a separate 500-instance 6-step validation set, not the reported held-out set | Keeps reported numbers free of selection bias |
| 2026-11-17 | `evaluate.py` moved into milestone 2 | The overfit exit check has to go through `solve` |
| 2026-11-17 | Early-stop patience only counts once validation accuracy ≥ 95% (`early_stop_min_accuracy`) | A k=1 pilot sat at 0% for its first 1,000 steps; the old rule would have stopped every run at step 3,000 untrained |
| 2026-11-17 | `solve` caps a walk at the node count (24), not 40 | No correct walk is longer; makes evaluation of untrained models cheaper |
| 2026-11-20 | Task shrunk to 16 nodes, train 4 steps, test 4/5/6/7/8 (was 24 nodes, 6, 6/8/10/12). Vocabulary still has 24 node tokens; 8 are unused | 24 nodes never left the loss plateau in the pilots; 8 nodes learned by step 900. Deviation from the second paper's exact graphs |
| 2026-11-20 | Default training settings: batch 512, cosine lr 3e-4 → 3e-5, weight decay 0.1, beta2 0.99, gradient clip 1.0 (was batch 64, constant 3e-4, 0.01, 0.999, no clip) | Old defaults reached 0% after 20,000 steps; these are close to the paper's code. Same for every k |
| 2026-11-20 | History conditions are k ∈ {1, 2, full}; main sweep is 36 runs | With 4-step training k ≥ 3 equals full |
| 2026-11-21 | Task shrunk again to 8 nodes, train 3 steps, test 3/4/5/6 (§1–§3 updated 2026-11-22) | Two 16-node k=1 pilots (N=16k and 160k, 5,000 steps) did not learn; 8 nodes is the only size confirmed to learn |
| 2026-11-21 | History conditions are k ∈ {1, full} only | With 3-step training, k ≥ 2 equals full during training. The study is now inductive vs. full, not a sweep over k |
| 2026-11-21 | Budget: max 4,000 steps, evaluate every 250 (was 20,000 / 500). Same for every k | The 8-node pilot learned by step 900; at batch 512 a 20,000-step failed run would take over an hour |
| 2026-11-21 | Caveat: 8 nodes has only about 80k distinct 3-step instances (40,320 YES + 40,320 NO) | N = 64k would cover most of the space, so the top training size needs rethinking before the main sweep |
| 2026-11-22 | Attention is measured at the `S` token that opens the target state, on contexts built from the oracle's states | That position predicts the next node. Oracle contexts let every step be measured even where the model stops early; flagged wherever the numbers are shown |
| 2026-11-22 | Main sweep sizes are N ∈ {250, 1k, 4k, 16k} (was 1k, 4k, 16k, 64k); 18 new runs started, approved by the user | 64k is most of the ~80k instance space; 250 adds a low end where sample efficiency can differ |
| 2026-11-17 | During training, log final accuracy on 200 held-out instances at each of 8/10/12 steps; each run saves `model.pt` | Shows how length generalization develops; never used for stopping. Weights are for the attention analysis |
| 2026-11-24 | Scale-up pilot approved by the user: 16 nodes, train 6 steps, test 6/8/10/12, k=1, one seed, fresh training instances every step (`--fresh`), up to 50,000 steps, laptop only. Output in `results/_pilot_16nodes_6step_fresh/` | The submitted proposal promised 24 nodes, 6 → 12 steps and four history conditions (full, 4, 2, 1); the 8-node version has only two. Four distinct conditions need 6-step training, and 12-step tests need ≥ 14 nodes. The second paper's own config for this experiment (`config/cycles_ood.py` in github.com/aryol/inductive-scratchpad) uses fresh samples, batch 512, cosine 3e-4 → 3e-5 over 20,000 iterations, a ~10M model (6 layers, 6 heads, width 384) and dropout 0.2; our earlier 16-node pilots stopped at 5,000 steps on fixed data. Fresh data is for learnability only: sample-efficiency runs still need fixed training sets. Timing: 0.53 s/step (970 examples/s), ~0.5 GB RAM, so ≤ ~8 h |
| 2026-11-24 | Still missing against the proposal, not yet approved: the position-reset test, examples/tokens-to-threshold (computable from the saved `history` of the 24 existing runs), and the middle history conditions (depend on this pilot) | Gap review against the proposal text on 2026-11-24 |
| 2026-11-24 | Pilot result: the 16-node / 6-step / k=1 fresh-data run learned. Validation trace accuracy was 0 until step ~8,500, 0.83 at 9,000 and ≥0.99 from 10,000; stopped early at 15,000 steps (117 min, 7.68M examples, 564M tokens). Held-out trace accuracy on 2,000 problems per length: 6 → 99.8%, 8 → 99.7%, 10 → 99.8%, 12 → 99.75%. No model or optimizer change was needed | The earlier 16-node failures came from the fixed training set and the short budget, not from model size. One seed only; whether full, 4 and 2 learn at this size, and whether a fixed training set of practical size works, is untested. Next runs need the user's approval |
| 2026-11-24 | Approved by the user and started: the other three history conditions at the pilot's size and settings (16 nodes, train 6, test 6/8/10/12, fresh data, seed 0, ≤ 50,000 steps each), run in the order full (`--k 0`), 4, 2. Same output folder; logs in `train_k0.log`, `train_k4.log`, `train_k2.log` | First look at the proposal's four conditions at 6 → 12 steps. Same model, optimizer, step budget, data stream and eval sets as the k=1 run, so only history differs. One seed each; fresh data, so this is not a sample-efficiency comparison |
| 2026-11-25 | Result of those three runs (seed 0, all stopped early). Held-out trace accuracy at 6/8/10/12 steps: full 100/0/0/0% (every failure is at the sixth state; final accuracy ≈ 50%); k=4 100/100/100/100%; k=2 100/100/100/100%; k=1 99.8/99.7/99.8/99.75%. First evaluation with validation trace accuracy ≥ 0.95 (evaluated every 500 steps): k=4 step 2,000 (1.02M examples, 80M tokens); k=2 step 2,500 (1.28M, 97M); full step 2,500 (1.28M, 100M); k=1 step 10,000 (5.12M, 376M) | At this size the middle conditions generalize like k=1 and reach the threshold about as fast as full history. A window of 2 or 4 states cannot tell step 5 from step 6, so the counting shortcut is unavailable. One seed: the ordering of time-to-threshold is not established, since plateau length varies by seed. More seeds need the user's approval |
| 2026-11-25 | Approved by the user and started: seeds 1 and 2 for all four conditions at 16 nodes (eight runs, same settings; logs `train_k<k>_s<seed>.log`). Added `scripts/threshold_table.py` (steps, examples and tokens to a validation trace-accuracy threshold, from the saved `history`). Attention measured on the four seed-0 models: best-head edge-attention share on 12-step problems is 0.99 (full), 0.99 (k=4), 1.00 (k=2), 0.39 (k=1); chance is 0.06 | Three seeds per condition for the proposal's main comparison. The k=1 model solves the task with its attention spread over several heads (two layer-3 heads at 0.39 and 0.33), unlike the 8-node k=1 models (0.99); one seed, unexplained so far |
| 2026-11-25 | Moved the 16-node runs to the Alliance Fir cluster at the user's request. Slurm array 6093788: 4 conditions × seeds 0, 1, 2, same settings as the laptop runs, one 10 GB H100 slice each, results in `results/_cluster_16nodes_6step_fresh/` on Fir (`~/scratch/cs886`). The laptop queue was stopped after six runs had finished (all of seed 0; k=4 and k=2 of seed 1); its full-history seed-1 run was cut off and left no result | Twelve runs in parallel instead of in sequence. All 12 are rerun on the cluster, including the six finished on the laptop, so the reported set comes from one machine and software version (torch 2.14.1, transformers 5.22.2 on Fir; 2.15.0 and 5.23.0 on the laptop). The laptop runs remain as a reproducibility check |
| 2026-11-25 | The user wants every reported number to come from one environment (Fir). The 8-node sweep (k ∈ {1, full} × N ∈ {250, 1k, 4k, 16k} × 3 seeds, default settings) was resubmitted there as array 6123940, results in `results/_cluster_8nodes_3step/`, with attention measured in the same job. Attention for the 16-node runs will be measured on Fir with `scripts/cluster/attention_job.sh` once the last run ends. Interim 16-node result, 11 of 12 runs: full history 0% trace accuracy at 8–12 steps in 3 of 3 seeds; k=4, k=2 and k=1 ≥ 99% at 6–12 in every finished run; steps to 95% validation: k=4 1,500/1,500/1,500, full 2,500/1,000/8,500, k=2 2,500/5,500/11,500, k=1 4,500/12,000/(seed 2 stalled near 30% by step 21,000, still running) | One machine and one software version for all reported results, so readers are not asked to reconcile two environments. The laptop results (24 + 6 runs) become an unreported cross-check; the paper and slides must be rewritten from the Fir numbers, which may differ from the laptop's |
| 2026-11-25 | All Fir runs finished and copied to `results/_cluster_16nodes_6step_fresh/` (12) and `results/_cluster_8nodes_3step/` (24), each with `attention.json` measured on Fir. **16 nodes, train 6, fresh data** (trace accuracy at 6/8/10/12; steps to 95% validation for seeds 0/1/2): full 100/0/0/0% in 3 of 3 seeds, 2,500/1,000/8,500; k=4 ≥ 99.85% everywhere, 1,500/1,500/1,500; k=2 100% everywhere, 2,500/5,500/11,500; k=1 ≥ 99.35% everywhere, 4,500/12,000/25,500 (seed 2 sat near 30% from step ~12,000 to ~25,000, then learned). **8 nodes, train 3, fixed N**: neither condition learns at N ≤ 1,000; at N=4,000 full learns in 3 of 3 seeds and k=1 in 2 of 3 (laptop: 1 of 3); at N=16,000 both learn in 3 of 3; full is 0% at 4–6 steps at every N, always failing at the third state. **Attention** (best head's edge share on the longest test length): 16 nodes — full 0.96/0.98/0.99; k=4 0.99/0.07/0.94; k=2 1.00/0.91/1.00; k=1 0.91/0.48/0.76. 8 nodes, N=16k — full 0.75/0.91/0.97; k=1 0.94/0.99/0.83 | The generalization result is the same in every seed at both sizes. Time-to-threshold is fastest and steadiest for k=4 and slowest for k=1, with wide seed spread for full, k=2 and k=1. The attention claim in the current paper ("the inductive model attends more sharply") does not hold on Fir: shares are not ordered by condition, and one k=4 model solves the task at 100% with no head above 0.13, so the correct-edge lookup need not happen at the position we measure. The paper and slides must be rewritten from these numbers |
| 2026-11-25 | Approved by the user and submitted on Fir as array 6173708: Experiment A at the proposal's 24 nodes (train 6, test 6/8/10/12, four conditions × seeds 0–2, fresh data, ≤ 50,000 steps, same model and optimizer; 12-hour limit per job). Results will be in `results/_cluster_24nodes_6step_fresh/` on Fir | The 16-node size was a laptop-era compromise; with the cluster the proposal's graph size can be run directly. Risk: the 4.2M model without dropout may not learn at 24 nodes (the paper uses ~10M with dropout 0.2); a larger model would need separate approval |
| 2026-11-25 | Timing on Fir: the 300-step smoke test took 43 s on a whole H100 against 193 s on a 10 GB slice (about 4.5×); the whole-card job waited 50 min in the queue, slices start within minutes. Data generation is only ~0.02–0.03 s of a ~0.5 s step, so the GPU, not the CPU, is the bottleneck (the earlier guess was wrong). On the user's approval, `train_array.sh` now defaults to a whole H100 and a 3-hour limit for future runs; the 24-node array already running on slices is left to finish | A full 50,000-step run at 24 nodes drops from ~7 h to ~1.5 h and fits the 3-hour queue class. Running jobs cannot be moved, and restarting them would save little |
| 2026-11-25 | On the user's instruction, the eight 24-node runs that had not learned (array indices 0, 2, 3, 6, 7, 9, 10, 11 of 6173708; index 1 had learned) were queued again on whole H100s as array 6210841, same settings and seeds, results in `results/_cluster_24nodes_6step_fresh_h100/`. Rule set by the user: a slice run is cancelled only if its whole-H100 twin starts within one hour of submission (18:08 MST); `scripts/cluster/swap_watch.sh` enforces this on the Fir login node and cancels any H100 task still pending after the hour | Whole H100s are ~4.5× faster but may wait in the queue; the slices are the fallback and must not be lost for a job that never starts. Caveat for reporting: 24-node results may then come from a mix of slices and whole cards of the same GPU model and software |
| 2026-11-26 | The one-hour hand-over ended with no whole-H100 task started; all 8 tasks of array 6210841 were cancelled by the watcher and no slice was stopped. Slice array 6173708 finished (all 12 COMPLETED) and was copied to `results/_cluster_24nodes_6step_fresh/`. **24 nodes, train 6, fresh data, ≤ 50,000 steps** (steps to 95% validation for seeds 0/1/2; `–` = never reached): full –/9,000/7,000; k=4 21,500/6,000/29,000; k=2 –/–/34,500; k=1 –/38,500/35,500. So 8 of 12 learned; k=4 is the only condition that learned in all three seeds. Held-out trace accuracy at 6/8/10/12 for learned runs: full s1 99.7/0/0/0, s2 100/0/0/0 (all 12,000 longer-problem failures at the sixth state, final accuracy ≈ 50%); k=4 ≥ 99.95% everywhere; k=2 s2 ≥ 99.85%; k=1 s1 ≥ 99.95%, s2 95.4–96.1%. Not learned: full s0 (0.1% at 6), k=2 s0 (0%), k=1 s0 (0%), k=2 s1 (12.6% at 6 steps, 0.4% at 12; still slowly improving at step 50,000). Differences from 16 nodes: full history reached the threshold sooner than k=4 in two seeds, so 'k=4 is cheapest' does not carry over; what carries over is that full never generalizes and limited history does. Confound to state: the cosine schedule ends at 3e-5 at step 50,000, so 'did not learn' means within this budget and schedule, not that the condition cannot learn. Attention job 6256206 submitted for this folder. A higher step limit or more seeds are not approved. |

## 10. Milestones 1–2 implementation plan (2026-11-17)

Milestone numbering follows `PROJECT_SPEC.md`. Schedule note: this starts in week 5 of §5,
so §5's dates no longer hold; milestones 3–4 will be re-scoped after the timing run.

### Step 0: setup
- [x] `uv venv --python 3.11 .venv`, then `uv pip install -e ".[dev]"` (torch 2.15, transformers 5.23, `mps` available).

### Milestone 1: the task (pure Python). Exit: `pytest` passes.
- [x] `task.py`: `cycle_edges`, `make_instance` (rule in §2, `ValueError` unless 2 ≤ L ≤ n − 2), `oracle_trace`, `make_dataset`, `instance_key`. `partition_into_cycles` removed.
- [x] `tests/test_task.py`: add cycle-structure tests for YES and NO, and the invalid-length test.
- [x] `tokens.py`: `encode_edge`, `encode_question`, `encode_state`, `parse_state`.
- [x] `formats.py`: `history_window`, `build_context`, `make_examples`.

### Milestone 2: tiny model + smoke test. Exit: overfits 100 instances to ~100% through `solve`, for k = 1 and full.
- [x] `data.py`: `ExampleDataset`, `collate` (labels −100 outside the target), `make_eval_sets` (disjoint from training), validation set.
- [x] `model.py`: Llama from `ModelConfig`; report the real parameter count.
- [x] `evaluate.py`: `greedy_decode`, batched `solve` on the model's own states, `score`, `evaluate`.
- [x] `train.py`: AdamW loop, same training graphs for every k, early stop, `config.json` + `metrics.json`.
- [x] Tests: label masking, eval/train disjointness, `solve` reuses a wrong state, `score` cases, model forward shape.
- [x] `scripts/smoke_overfit.py`: the exit check.
- [x] Timing: ~200 steps at k = 1, N = 16k, extrapolated. No full run without approval.

### Results (2026-11-17, Apple `mps`)
- `pytest`: 48 passed.
- Parameter count: 4,214,016 (§2 says ~3M; the real figure for 4 layers / 4 heads / width 256 is ~4.2M).
- Overfit check, 100 instances, solved through `solve`: k = 1 reached 100% trace accuracy by step 750;
  full history by step 500. No format failures.
- Timing at batch 64: 0.075 s/step (k = 1), 0.080 s/step (full). A run at the full 20,000-step budget is
  about 25–27 min of training plus about 5 min of evaluation, so at most ~30 min per run; early stopping
  should make most runs shorter. Worst case: milestone 3 (2 conditions × 3 seeds) ≈ 3 h; the 48-run sweep ≈ 24 h.
