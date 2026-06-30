# ProteinSynergyDock

Structure-aware drug combination synergy prediction with real molecular docking, cell-line context, and resistance mutation analysis.

[![Tests](https://github.com/Aprameya05/ProteinSynergyDock-App/actions/workflows/tests.yml/badge.svg)](https://github.com/Aprameya05/ProteinSynergyDock-App/actions/workflows/tests.yml)
[![Live App](https://img.shields.io/badge/demo-live-green)](https://proteinsynergydock-app-kddtbdmnkixw9c8jfnf8un.streamlit.app/)

## What this is

A GNN-based model that predicts whether a pair of drugs will act synergistically against a specific cancer cell line, grounded in real AutoDock Vina docking against the target protein structure rather than relying on chemical fingerprints alone. Trained on 107,103 synergy measurements from the NCI ALMANAC dataset across 60 cancer cell lines.

This isn't a toy demo — every prediction in the "Predict Synergy" tab runs an actual docking simulation against a live-fetched PDB structure, and every drug's SMILES string is validated against RDKit before being trusted (see `tests/test_chemistry.py` — this exists because two malformed SMILES strings shipped to production undetected until a user hit them at runtime, which is exactly the failure class this suite now catches in CI).

## Architecture

```
core.py          Pure business logic — zero Streamlit dependency, fully unit tested
app.py           Streamlit UI layer — imports from core.py, no duplicated logic
tests/           202 tests covering chemistry validity, docking geometry,
                  synergy verdict thresholds, mutation database integrity,
                  and the natural-language query parser
.github/workflows/  CI: tests run on every push across Python 3.10 and 3.11
```

The split between `core.py` and `app.py` matters: Streamlit scripts execute top-to-bottom on every interaction and can't be imported for testing in the normal sense. Pulling all the actual logic — SMILES parsing, docking box calculation, synergy verdict thresholds, the NL query parser — into a Streamlit-free module means it can be tested directly, and `app.py` is reduced to what it should be: a thin rendering layer.

## Model

**ProteinSynergyDockV2** — a GATv2-based graph neural network:
- Each drug is encoded as a molecular graph (atoms as nodes, bonds as edges) via a 2-layer GATv2 encoder
- Cross-drug attention fuses the two drug embeddings
- FiLM conditioning injects Gene Ontology pathway context
- Real AutoDock Vina docking scores (not just structural features) feed into the final prediction head
- Cell-line embeddings let the same drug pair predict differently across 60 cancer types

Trained checkpoint: Pearson r and AUROC reported live in the app sidebar, computed on a held-out split of NCI ALMANAC.

## Features

| Tab | What it does |
|---|---|
| Predict Synergy | Real-time docking + synergy prediction for any drug pair against any PDB structure |
| Synergy Landscape | Full heatmap of precomputed pairwise synergy across a cancer panel |
| Cell Line Comparison | How one drug pair's synergy varies across all 9 cancer panels |
| Clinical Trials | Live ClinicalTrials.gov search for a given drug combination |
| Literature | Live PubMed search for supporting publications |
| Drug Repurposing | Best synergy partners for a given anchor drug |
| Mechanism Explorer | Pathway/target-based rationale for why a combination should or shouldn't synergize |
| Resistance Mutations | Binding affinity delta between wild-type and mutant protein for known resistance variants (e.g. EGFR T790M, BRAF V600E) |
| 4D Trajectory | Simulated binding approach trajectory with energy profile |
| Natural Language Query | Rule-based parser over precomputed synergy data — no API key required |
| Polypharmacology Network | Systems-level view of drug-pathway relationships across the full panel, not just pairwise |

## Running tests

```bash
pip install -r requirements.txt
pip install pytest
pytest tests/ -v
```

202 tests, covering:
- Every drug's SMILES string parses to a valid molecule (regression test for the SMILES incident above)
- Docking binding-box calculation correctly excludes water molecules from the ligand centroid
- Synergy verdict threshold boundaries are exact and documented
- Every drug referenced in the resistance mutation database is actually selectable in the UI
- The natural language query parser doesn't crash on empty/gibberish input and correctly filters by cancer type and drug mentions

## Known limitations

- Synergy verdict thresholds (`>0.5` strong, `>0.1` mild, `>-0.1` additive) are heuristic cutoffs, not derived from a calibration study — documented as such in the app's "How to interpret" expander
- Predictions are point estimates with no uncertainty quantification yet (see Roadmap)
- Docking uses a single AutoDock Vina run per drug rather than ensemble pose averaging

## Roadmap

- [ ] Monte Carlo dropout for prediction uncertainty (mean ± std instead of point estimate)
- [ ] Benchmark against published synergy models (DeepSynergy, MatchMaker) on identical held-out splits
- [ ] Dose-response synergy surface prediction (Bliss/Loewe) instead of single-point synergy score

## Links

- [Live app](https://proteinsynergydock-app-kddtbdmnkixw9c8jfnf8un.streamlit.app/)
- [Model repo](https://github.com/Aprameya05/ProteinSynergyDock)
- Related: [ProteinWhisper](https://github.com/Aprameya05/ProteinWhisper) (zero-shot protein function annotation), [DrugSynergy3D](https://github.com/Aprameya05/DrugSynergy3D) (SE(3)-equivariant synergy GNN)
