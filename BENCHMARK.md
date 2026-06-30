# ProteinSynergyDockV2 — Benchmark Context

## Headline result

**The model retains most of its predictive power on cancer cell lines it has never seen during training.** Pearson r drops only 0.10 (from 0.567 to 0.466) when evaluated under a strict cell-line-held-out split instead of a random split — a modest, well-characterized generalization gap rather than a collapse, indicating the model has learned transferable structure-synergy relationships rather than memorizing cell-line-specific patterns.

## Two evaluation protocols, both real, both reported

| Evaluation | Pearson r | AUROC | Test set | Cell lines in test |
|---|---|---|---|---|
| **Random split** (original training run, epoch 82) | 0.567 | 0.795 | ~21,400 rows, randomly sampled | All 60 cell lines also appear in training |
| **Cell-line-held-out split** (this experiment) | 0.466 | 0.755 | 20,903 rows | 12 cell lines, zero overlap with the 48 training cell lines |

**Generalization gap: 0.100 Pearson r, 0.040 AUROC.**

### What each protocol actually tests

The random split shuffles all ~107K (drug A, drug B, cell line) rows and randomly assigns 80/20 to train/test. Every cell line appears in both sets — the model has seen synergy data for all 60 cancer cell lines during training, just not that exact drug-pair-cell-line triplet. This is the easier evaluation and is what most published synergy models report by default.

The cell-line-held-out split instead removes 12 of the 60 cell lines (20%) **entirely** from training — `786-0, CAKI-1, HCT-15, HL-60(TB), NCI-H522, OVCAR-4, SK-MEL-2, SK-MEL-28, SK-OV-3, SNB-19, T-47D, UACC-257` — covering renal, colon, leukemia, lung, ovarian, melanoma, CNS, breast, and other cancer types. The model trained on 77,607 rows from the remaining 48 cell lines, was validated on 8,593 held-out rows from those same 48 cell lines (for early stopping / model selection only), and was tested exclusively on 20,903 rows from the 12 cell lines it never saw a single synergy measurement for during training. This is a meaningfully harder, more clinically relevant test: can the model predict synergy for a cancer type it has zero training signal for, relying purely on the drug structures, docking scores, and GO pathway context generalizing across cell lines?

### Why a 0.10 gap is a genuinely good result

A drop from random-split to held-out-split performance is universal and expected in this literature — every synergy model shows some gap, because random splits are inherently easier. What varies is the *size* of the gap, and that size is itself informative: a small gap means the model is learning biology that transfers; a large gap means it's mostly learning cell-line-specific shortcuts that don't generalize. A 0.10 Pearson r gap (18% relative drop) on a 20%-of-cell-lines holdout is a modest, defensible result that supports treating this model as having learned something beyond simple memorization, without overclaiming it as fully solved generalization (it is not — performance is still measurably lower on unseen cell lines, and that's reported honestly above, not hidden).

## Comparison against published NCI-ALMANAC literature

The closest legitimate external comparison remains **Sidorov et al. (2019), "Predicting Synergism of Cancer Drug Combinations Using NCI-ALMANAC Data"** (Frontiers in Chemistry) — same dataset family, per-cell-line Random Forest / XGBoost models, Pearson r ranging 0.43–0.86 depending on cell line.

| Model | Dataset | Split | Pearson r |
|---|---|---|---|
| **ProteinSynergyDockV2 — random split** | NCI-ALMANAC | Random | **0.567** |
| **ProteinSynergyDockV2 — cell-line-held-out** | NCI-ALMANAC | Held-out (12/60 cell lines) | **0.466** |
| Sidorov et al. RF/XGBoost | NCI-ALMANAC | Per-cell-line | 0.43 – 0.86 (range) |

Both of this work's results fall within Sidorov et al.'s reported range, even under the harder held-out evaluation — which is a stronger, more defensible claim than the random-split number alone, since it demonstrates the result isn't an artifact of evaluation leniency.

Numbers from MatchMaker (r=0.79, but on DrugComb, not NCI-ALMANAC) and DeepSynergy (AUROC ranging 0.666–0.844 across different independent replications on NCI-ALMANAC) are still excluded from direct comparison for the same dataset/split-mismatch reasons documented below — that reasoning is unchanged by this experiment.

## Why other published numbers still aren't included

Synergy prediction papers report substantially different numbers for the same model depending on three factors that are rarely held constant across papers: which dataset (NCI-ALMANAC vs. DrugComb vs. Merck-2016 — different drugs, cell lines, and synergy score definitions), which split strategy (random vs. cell-line-held-out vs. drug-held-out vs. leave-triple-out), and which paper is reporting (independent replications of the same architecture can differ by 15+ AUROC points). Mixing these into one table would be comparing different experiments, not different models. This work now reports both a random-split and a held-out-split number specifically so its own evaluation rigor is transparent and matched as closely as possible to what a careful external comparison would require.

## How to present this honestly (in a CV, paper, or interview)

**Defensible:** *"On NCI-ALMANAC, the model achieves Pearson r=0.567 under a random train/test split and r=0.466 under a stricter cell-line-held-out split (12 of 60 cell lines fully excluded from training), demonstrating a modest 0.10 generalization gap consistent with the model learning transferable structure-synergy relationships rather than cell-line-specific memorization. Both results fall within the range reported by prior NCI-ALMANAC-specific synergy models (Sidorov et al., 2019: r=0.43–0.86 depending on cell line)."*

**Also defensible, shorter:** *"The model generalizes to entirely unseen cancer cell lines with only a modest performance drop (Pearson r: 0.567 → 0.466), suggesting it captures real structure-synergy relationships rather than memorizing per-cell-line patterns."*

**Not defensible:** *"Our model outperforms state-of-the-art methods like MatchMaker and DeepSynergy."* (different datasets, unverified split parity — this claim is not supported by anything in this document and should not be made.)

## Reproducing this result

The held-out evaluation can be reproduced via `kaggle_heldout_retrain_v2.py` in this repository — built directly from the original training notebook with only the train/test split logic changed (random shuffle → cell-line holdout), every other hyperparameter (architecture, optimizer, loss weights, batch size, epoch count) identical to the original run that produced the deployed checkpoint. Full results, including the exact 12 held-out cell lines, are saved to `heldout_results.json` after each run for verification.
