# Evaluation

* **Loss** = mean token negative log-likelihood in nats (natural log), token-weighted across batches.
* **Perplexity** = `exp(loss)`: the effective number of equally-likely tokens the model is choosing among. With log base 2
  the same quantity is `2^{bits/token}`; **bits/token** = `loss/ln 2`. Uniform guessing gives PPL = V (initial loss ≈ ln V,
  checked in tests and in the first validation line of every run).
* Perplexity here is **per BPE token of this tokenizer**; it cannot be compared with word-level WikiText numbers or with
  other vocabularies (a bigger vocab means fewer, harder-to-predict tokens).
* Non-overlapping windows: the first tokens of each window have little context, so reported PPL is slightly pessimistic
  versus a sliding-window evaluation.

## Splits (no leakage)

| dataset | train | validation | test |
|---|---|---|---|
| wikitext2 | official `train.txt` | official `valid.txt` | official `test.txt` |
| tinyshakespeare / custom | first 90 % of paragraphs | next 5 % | last 5 % |

Contiguous splitting (no shuffling) keeps neighbouring, near-duplicate passages in the same split. The tokenizer is
trained on the train split only. Periodic evaluation uses `eval_iters` validation batches (a fixed prefix of the split →
comparable across steps); the end-of-run numbers use the **full** validation and test splits. `best.pt` is chosen on
validation loss; the test set is only read for the final report.

## Fixed qualitative prompts

`evaluation/evaluation.py::EVAL_PROMPTS` (6 prompts). At steps listed in `training.sample_steps` (default 0, 1000, 5000)
and at the end, each prompt is continued with (a) greedy decoding + repetition penalty 1.1 and (b) a seeded sample
(T=0.8, top-k 50, seed 1234), saved to `runs/<run>/samples/step_<N>.txt`, so snapshots are directly comparable.

## Scripts

`python scripts/evaluate.py --checkpoint checkpoints/best.pt` → train-subset / validation / test loss, PPL, bits/token
in `outputs/metrics/eval_<ckpt>.json`.
