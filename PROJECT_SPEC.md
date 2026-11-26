# How Much Scratchpad History Do Transformers Need to Learn and Generalize?

CS886 course project. Status: direction revised 2026-10-20; no code yet.

This replaces the earlier "Verified Inductive Scratchpads" spec. Removed: the CS846
assumption, the online checker, rejection-and-repair, correctness certification, and
the frozen-API-model requirement. This is a **train-from-scratch** study.

## Source papers

- **P1:** "From Sparse Dependence to Sparse Attention" (arXiv 2410.05459). Question: does
  intermediate supervision make a transformer learn from fewer examples, and how does
  attention learn to retrieve the relevant inputs? Theory is for parity, under specific
  training assumptions. Its conclusion names tasks beyond parity as future work.
- **P2:** "How Far Can Transformers Reason? The Globality Barrier and Inductive
  Scratchpad" (arXiv 2406.06467). Question: can a model trained on short executions
  run the procedure for more steps? Its inductive scratchpad lets each new state see
  only the question and the previous state (attention mask + position re-indexing, or
  equivalently state-pair training examples).

P1's parity theorem does **not** automatically apply to the graph task. Anything we
show here is empirical.

## Research question

How does the amount of previous scratchpad history visible to the model affect
(1) how fast it learns, (2) what it attends to, and (3) whether it solves longer
problems than it was trained on?

## Task: cycle reachability (from P2)

Input: a list of directed edges over 24 nodes, plus a query "can node `s` reach node `t`?".
The scratchpad follows edges one step at a time; each state records the current node
(plus whatever else the next update needs, e.g. the start node to detect a full loop).
The answer is read from the final state.

- Train: instances needing 6 traversal steps.
- Test: instances needing 12 traversal steps, same 24 nodes, no extra training.

The node count stays fixed, so only reasoning length changes.
Verified against P2 §3.2.1 on 2026-11-17. For an L-step instance, NO is two cycles of length
L and 24 − L (source in the L-cycle) and YES is one 24-cycle with the target L ahead. Our
state format and node naming differ from P2; see `plan.md` §9.

## Conditions (training *and* evaluation, not just prompting)

Every condition uses the same graphs and the same state format. Only what the model can
attend to when it writes the next state changes.

| Condition | Visible when predicting state *i* | Notes |
|---|---|---|
| Full scratchpad | question + states 1..i-1 | P2 baseline |
| Recent-k | question + last k states, k ∈ {2, 4, 8} | the new sweep |
| Inductive | question + state i-1 only (k = 1) | P2's method |

The state is Markov: it holds everything the next update needs. So restricting history
removes **redundant** context, not useful information. This is what makes the sweep fair.

Implementation: one shared model and data pipeline. History is restricted with an
attention mask, and positions are re-indexed so state *i* always "looks like" it sits
right after the question. Check P2's description of re-indexing before implementing it.

## Measurements

1. **Learning efficiency.** Number of training examples to reach a target accuracy
   (e.g. 95%) on held-out 6-step problems. Also log training tokens and FLOPs, because
   longer history costs more per example.
2. **Length generalization.** Accuracy on 12-step problems (and maybe 8 and 10 for a
   curve), with no further training. Report both final-answer and per-step accuracy.
3. **Attention behaviour.** For each update, the "correct" input tokens are known: the
   edge whose source is the current node. Measure the attention mass placed on that edge,
   per head and layer. Do **not** use lower attention entropy alone as success. Masking
   history already lowers entropy mechanically.

Run several seeds per condition; report means with spread.

## Contribution (what is new and what isn't)

- Full vs. inductive alone mostly reproduces P2.
- New: a controlled sweep over history length k. Does the setting that learns fastest
  also generalize furthest? How does retrieval of the right edge change with k?
- Also an empirical extension of P1's sample-efficiency/attention view beyond parity.

This is a candidate contribution, not a claim that nobody has studied it.

## Confounds to watch

- Full history at test time creates sequences longer than any seen in training. That is
  part of what we measure, but report sequence length next to the results.
- Recent-k with k ≥ 6 equals full history during training (only 6 states). It differs
  only at test time. Pick k values that make sense given this.
- Equalize model size, optimizer and step budget across conditions. Report results both
  per example and per token.

## Role of the reference program

A plain Python program generates graphs, correct traces and labels, and scores outputs.
It never corrects the model during testing.

## Milestones

1. Pure-Python generator, oracle trace, tokenizer, scorer, plus unit tests.
2. Small decoder-only transformer with a configurable history mask; a smoke test on tiny data.
3. Full vs. inductive replication on 6→12 steps.
4. k sweep, sample-efficiency curves, attention analysis.

Get approval for model size and compute budget before milestone 3.
