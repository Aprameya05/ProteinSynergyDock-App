# ProteinSynergyDock

Predict whether two cancer drugs will work better together. The prediction is grounded in real molecular docking geometry, not just chemical string matching. Results come with uncertainty estimates, pharmacological analysis, and output formatted as a FHIR R4 healthcare interoperability resource.

**Live app:** [proteinsynergydock-app-kddtbdmnkixw9c8jfnf8un.streamlit.app](https://proteinsynergydock-app-kddtbdmnkixw9c8jfnf8un.streamlit.app/)

**FHIR API:** [proteinsynergydock-fhir-api.onrender.com/docs](https://proteinsynergydock-fhir-api.onrender.com/docs)

**Tests:** ![Tests](https://github.com/Aprameya05/ProteinSynergyDock-App/actions/workflows/tests.yml/badge.svg)

---

![3D docking visualization](demo.png)

*Vemurafenib (cyan) and Trametinib (orange) docked inside BRAF kinase (PDB: 3OG7). FDA-approved combination for BRAF V600E melanoma.*

---

![Architecture](architecture.png)

---

## Why this exists

Most synergy prediction tools take two SMILES strings and output a number. They treat molecules as text. This misses the most important physical question: where do the two drugs actually bind on their shared target protein, and does that geometry make combination therapy sensible?

Two drugs that bind the same ATP pocket compete with each other. Two drugs that hit separate binding sites on the same protein, or hit different proteins in the same oncogenic pathway, can produce effects stronger than either drug alone. Getting that geometric and pathway-level context right is the whole point.

This app combines AutoDock Vina molecular docking, a graph neural network drug encoder, and cell-line-specific cancer biology to answer the question properly.

---

## What happens when you run a prediction

1. You provide Drug A, Drug B (SMILES strings), a cancer cell line, and a target protein PDB ID.

2. The app fetches the crystal structure from RCSB PDB using the provided ID.

3. AutoDock Vina runs independently for each drug: it searches the protein pocket for the lowest-energy binding pose and returns a binding affinity in kcal/mol. This is real docking, not a lookup.

4. RDKit generates a 3D conformer from each drug's SMILES. The conformer is converted into a molecular graph where each atom is a node with a 7-dimensional feature vector (atomic number, formal charge, hybridization, aromaticity, H-bond donor/acceptor status, ring membership) and each bond is an edge.

5. A GATv2 graph attention network (Graph Attention Network v2, Brody et al. 2022) encodes each drug graph into a 256-dimensional embedding. Four attention heads capture different structural patterns across two message-passing layers.

6. A cross-drug attention module takes both drug embeddings and runs multi-head self-attention across the pair. This lets the model learn which structural features of Drug A are relevant given the structure of Drug B, and vice versa.

7. The protein's biological function is encoded via a GO (Gene Ontology) context vector. A FiLM (Feature-wise Linear Modulation) layer uses this vector to condition the fused drug representation, scaling and shifting it based on what the protein actually does.

8. A learned embedding for the specific cancer cell line (from the NCI-60 panel) is concatenated with the fused drug+protein representation and the two docking affinities.

9. The prediction head (three linear layers with LayerNorm and Dropout) outputs a synergy score and a confidence value.

10. Monte Carlo Dropout runs 20 stochastic forward passes with dropout active at inference time (Gal & Ghahramani 2016). The mean gives the final prediction; the standard deviation gives calibrated uncertainty. If the model is uncertain, you can see it.

11. Both docked drug poses render inside the protein's 3D surface in the browser using py3Dmol.

12. The result is serialized as a FHIR R4 DiagnosticReport resource with embedded Observation components.

---

## The model

**ProteinSynergyDockV2** with approximately 1.87 million parameters.

| Component | Details |
|---|---|
| Drug encoder | GATv2Conv, 2 layers, 4 attention heads, 128 hidden, 256 output |
| Cross-drug attention | MultiheadAttention (4 heads) over the [Drug A, Drug B] token pair |
| Protein conditioning | FiLM layer over 512-dim GO embedding |
| Cell line embedding | nn.Embedding, 60 NCI-60 lines, 32-dim |
| Prediction head | Linear(768, 512) + LayerNorm + ReLU + Dropout(0.2) + Linear(512, 256) + ReLU + Dropout(0.1) + Linear(256, 2) |
| Uncertainty | Monte Carlo Dropout, 20 forward passes |

Training data: 107,103 drug-drug-cell-line triplets from the NCI ALMANAC dataset, with real AutoDock Vina docking scores computed for 842 drug-target pairs across 20 cancer-relevant proteins.

| Metric | Value |
|---|---|
| Pearson r (held-out test set) | 0.5667 |
| AUROC (synergistic vs. not) | 0.7946 |

Full evaluation methodology and per-cell-line breakdown: [`heldout_results.json`](heldout_results.json) and [`BENCHMARK.md`](BENCHMARK.md).

---

## 17 analysis tabs

### 1. Predict Synergy
The main prediction tab. Runs the full pipeline described above. After prediction, it shows the Tanimoto structural similarity between the two drugs (ECFP4 Morgan fingerprints) with a warning if similarity is high (>0.65), since very similar drugs tend to compete rather than synergize. A quick ADMET snapshot appears inline: MW, LogP, QED, solubility, BBB penetration, and Lipinski compliance for both drugs.

### 2. Synergy Landscape
Precomputed synergy heatmaps across 28x28 drug pair matrices for 9 cancer panels. Shows which drug combinations are consistently predicted as synergistic across the landscape, not just for a single pair you asked about.

### 3. Cell Line Comparison
Take the same drug pair and compare predicted synergy across different cancer cell lines. BRCA, LUAD, and COAD lines often show dramatically different synergy profiles for the same combination.

### 4. Clinical Trials
Searches ClinicalTrials.gov for active and recruiting trials involving your predicted drug combination. Bridges the gap between a computational prediction and the clinical reality of what is actually being tested in humans.

### 5. Literature
PubMed mining for publications about your specific drug pair and cancer type. Returns real citation metadata, not a hallucinated summary.

### 6. Drug Repurposing
Screens the drug library for non-obvious high-synergy partners with a query drug. Uses precomputed synergy matrices to surface drugs that might not be the standard-of-care partner but score highly.

### 7. Mechanism Explorer
Pathway-level rationale for why a drug pair is predicted to synergize or antagonize. Uses the SYNERGY_RULES lookup (curated from literature) to explain predictions in terms of biological pathways: BRAF+MEK, PI3K+mTOR, EGFR+VEGFR, etc.

### 8. Resistance Mutations
Applies known clinical resistance mutations to the prediction. BRAF V600E, EGFR T790M, ALK L1196M, BCR-ABL T315I and others. Shows how predicted synergy changes when the tumor has acquired resistance.

### 9. 4D Trajectory
Time-evolved docking trajectory animation. Visualizes how docked drug poses shift as the protein undergoes conformational sampling, adding a dynamic dimension beyond the single static pose from docking.

### 10. Query
Natural-language interface over precomputed synergy data. Ask questions like "which drug pairs are synergistic in BRAF-mutant melanoma" and get answers from the actual data.

### 11. Polypharmacology Network
Systems-level drug-protein-pathway interaction graph. Shows each drug's target network, overlapping targets between the two drugs, and which pathway nodes each drug hits. Useful for understanding the systems biology context of a combination.

### 12. Clinical Interop (FHIR)
Converts the live prediction into a spec-compliant FHIR R4 DiagnosticReport. The report includes Observation components for the synergy score, synergy probability, confidence interval, and docking affinities. Input validation returns a proper FHIR OperationOutcome on error rather than a Python stack trace. Every prediction (successful or rejected) is recorded in an append-only audit log with cryptographic hash chaining.

### 13. ADMET Analysis
Full pharmacokinetic and pharmacodynamic profile for both drugs, computed entirely from SMILES using RDKit. No database lookups, no cached values.

Properties computed:
- **Physicochemical**: MW, LogP, HBD, HBA, TPSA, rotatable bonds, ring count, aromatic rings, heavy atom count, Fsp3, stereocenters, halogens
- **Drug-likeness**: QED score (RDKit), Lipinski Rule of 5 (with per-rule breakdown and violation count), Veber bioavailability rules (rotatable bonds <=10, TPSA <=140), composite drug-likeness score
- **Solubility**: ESOL estimated aqueous solubility using the Delaney (2004) equation: log S = 0.16 - 0.63 * cLogP - 0.0062 * MW + 0.066 * rotatable bonds - 0.74 * aromatic proportion. Classified as Poor / Moderate / Good.
- **CNS penetration**: Blood-brain barrier estimate using the Clark model (scored 0-4 based on TPSA <90, MW <400, 0<=LogP<=5, HBD <3). Score >=3 predicts penetration.
- **Transport**: P-glycoprotein substrate heuristic (MW >400 and TPSA >60), CYP3A4 substrate heuristic, CYP2D6 inhibitor heuristic
- **Pharmacophore counts**: H-bond donors, H-bond acceptors, aromatic rings, hydrophobic centers, basic nitrogens, acidic groups, halogens, stereocenters, amide bonds, sulfonamides (via SMARTS pattern matching)

Displayed as a radar chart comparing both drugs across 8 normalized dimensions, and as a side-by-side property table.

### 14. Chemical Space
Positions your two query drugs within the chemical space of all 35+ drugs in the library.

Method: Morgan fingerprints (ECFP4, radius=2, 1024 bits) computed via RDKit for every drug in the library. PCA (scikit-learn, n_components=2) reduces the fingerprint matrix to 2D. The scatter plot shows where each drug sits, with query drugs highlighted as stars. Variance explained by each principal component is shown on the axes.

Also computes the full pairwise Tanimoto similarity matrix for all drugs and renders it as an interactive heatmap. Tanimoto >= 0.85 means near-identical scaffolds. Below 0.4 means genuinely diverse chemistry.

Nearest-neighbor lookup: for each query drug, finds the most structurally similar drugs in the library (by Tanimoto) and shows their known synergy profiles.

### 15. Combination Index
Quantitative synergy classification using two independent mathematical models.

**Chou-Talalay Combination Index (Chou 2010, Cancer Res 70:440)**: CI < 1 = synergy, CI = 1 = additivity, CI > 1 = antagonism. CI is approximated from the GNN synergy score as CI = exp(-synergy_score). This mapping preserves the direction and relative magnitude of the GNN output while placing it on the standard pharmacological CI scale.

**Bliss Independence model (Bliss 1939)**: converts synergy scores to effect probabilities via sigmoid, then computes the expected combined effect under the independence assumption (E_AB_expected = E_A + E_B - E_A * E_B). The difference between observed and expected combined effect gives the Bliss deviation: positive = synergy, negative = antagonism.

**Hill equation dose-response curves**: for each drug, plots Effect(C) = C^n / (EC50^n + C^n) x 100% over three log decades of concentration. EC50 is estimated from the drug's TPSA and LogP (low TPSA + moderate LogP correlates with better permeability and lower effective concentration).

**Combination dose-effect matrix**: 8x8 matrix showing predicted combined effect across a grid of Drug A and Drug B concentrations. Cells are colored by Bliss-predicted effect, shifted by the GNN synergy score to show where synergy is expected to be strongest.

**CI sensitivity sweep**: plots CI as a function of synergy score from -2 to +2, showing where your current prediction lands on the full range.

### 16. Explainability
Per-atom importance scoring for both drug molecules. Uses RDKit chemistry rules to assign importance to each atom based on its pharmacophoric role:

- Aromatic heteroatoms (N, O, S in ring systems): highest importance, often key for binding
- H-bond donors and acceptors: important for protein pocket interactions
- Charged atoms: high importance, often involved in salt bridges
- Ring heteroatoms: contribute to scaffold binding
- Aliphatic carbons: lowest importance unless at branch points

Importance scores are aggregated into feature classes and shown as normalized bar charts for each drug. The structural overlap score (Tanimoto) and a mechanism-based synergy rationale from the SYNERGY_RULES database explain why the model expects the specific combination to work or not.

This tab is intentionally transparent: it does not use gradient-based saliency (which requires backpropagation through the GNN and can be misleading) but instead uses interpretable chemistry rules that a medicinal chemist can verify directly.

### 17. Report
Generates a complete standalone HTML report from the last prediction. The report includes: synergy score and uncertainty, Chou-Talalay CI, Bliss classification, docking affinities, full ADMET table for both drugs, interpretation guide, model metadata, and a disclaimer. The HTML is self-contained (no external dependencies) and renders correctly in any browser. Also provides CSV export of the full prediction history for the session and JSON export of the raw prediction data.

---

## FHIR clinical interoperability layer

Most drug-discovery ML tools stop at a prediction number. Real clinical software has to expose predictions in a format that EHR platforms can consume, validate inputs against actual clinical data schemas, and maintain an auditable record of all predictions.

**FHIR R4 resources** (`core_fhir.py`): predictions are serialized as `DiagnosticReport` with embedded `Observation` components. Invalid inputs (bad SMILES, unrecognized cell line) return a spec-correct `OperationOutcome` resource instead of a Python exception. This is the same resource format used by Oracle Health (Cerner), Epic, and other major EHR platforms.

**Hash-chained audit log** (`audit_log.py`): every prediction, including rejected ones, is written to an append-only log. Each entry is SHA-256 hashed together with the previous entry's hash, creating a chain. The `verify_chain()` function detects any modification to any historical record.

**Input validation**: cell line identifiers are checked against the actual NCI-60 nomenclature. PDB IDs are checked against RCSB before a docking job starts. Malformed SMILES are caught by RDKit before they reach the model.

**Public REST API** (`api.py`, deployed on Render): `POST /fhir/DiagnosticReport` runs live model inference and returns FHIR JSON. Interactive Swagger docs at [proteinsynergydock-fhir-api.onrender.com/docs](https://proteinsynergydock-fhir-api.onrender.com/docs). Note: the API does not run Vina docking (too slow for a synchronous HTTP endpoint) -- `docking_affinity` is omitted from API responses unless supplied directly in the request body.

---

## Quick-start examples

All SMILES below are pre-loaded in the Quick Examples dropdown.

| Drug A | Drug B | PDB Target | Expected result | Reason |
|---|---|---|---|---|
| Vemurafenib | Trametinib | 3OG7 (BRAF) | Strongly synergistic | BRAF inhibitor + MEK inhibitor; FDA-approved combination for BRAF V600E melanoma. Vertical pathway blockade. |
| Imatinib | Dasatinib | 2HYY (ABL1) | Antagonistic | Both drugs compete for the ABL1 ATP-binding pocket. Same binding site = competition. |
| Erlotinib | Lapatinib | 1IVO (EGFR) | Synergistic | Erlotinib blocks EGFR extracellular; Lapatinib targets a different binding mode. Complementary inhibition. |
| Olaparib | Rucaparib | 4DQY (PARP1) | Mildly synergistic | Both are PARP1 inhibitors but with different PARP-trapping efficiency profiles. |

---

## Codebase structure

```
app.py              Streamlit UI, 17 tabs, all display logic
core.py             Model definitions, docking helpers, drug data, synergy rules
admet_utils.py      ADMET computation, Tanimoto similarity, Bliss/CI models, report generator
core_fhir.py        FHIR R4 resource construction (DiagnosticReport, Observation, OperationOutcome)
audit_log.py        Hash-chained append-only prediction audit log
model_bridge.py     Thin adapter between core.py and the FHIR API layer
api.py              FastAPI REST API, deployed separately on Render
tests/              266 automated tests covering all layers
```

---

## Engineering notes

**Separated business logic**: `core.py` has zero Streamlit imports. The model, docking helpers, and synergy rules are all testable in isolation. `app.py` is UI-only. This means CI runs the real prediction code, not a mock.

**266 automated tests** run on every push via GitHub Actions across Python 3.10 and 3.11. Tests cover: model forward pass with real graph inputs, SMILES validation for all 35+ drugs, FHIR resource schema correctness, audit log chain integrity, API endpoint contracts, ADMET property edge cases (invalid SMILES, single-atom molecules, very large molecules).

**No hardcoded property values**: every ADMET number, every Tanimoto score, every pharmacophore count is computed from the SMILES at runtime via RDKit. There is no lookup table of pre-computed drug properties.

**All 35+ drug SMILES validated on every CI run**: if a SMILES fails RDKit parsing, the CI run fails. No invalid molecules can enter the drug library through a merge.

---

## Known limitations

**Prediction accuracy**: held-out Pearson r of 0.5667 reflects the genuine difficulty of synergy prediction. Most synergy signal comes from learned chemical and biological priors, not from per-pair docking geometry, because real docking data is expensive and the 842 Vina runs cover only a fraction of training pairs.

**GO embeddings**: the GO context vector used during inference is a fixed-size placeholder, not a per-protein computed embedding. Per-protein GO embeddings (from ProteinWhisper) improve predictions but require the embedding to be computed for each new target.

**API docking**: the public Render API does not run AutoDock Vina (Vina runs take 30-90 seconds per drug, too slow for a synchronous endpoint). Docking affinities are excluded from API responses unless you supply them directly.

**Not for clinical use**: this is a research tool. Predictions have not been validated against clinical outcomes and are not reviewed or approved for any medical application.

---

## Stack

| Component | Technology |
|---|---|
| Molecular docking | AutoDock Vina 1.2.7 + OpenBabel |
| Drug graph encoding | RDKit (graph construction, conformers, ADMET), PyTorch Geometric (GATv2) |
| Machine learning | PyTorch |
| Interoperability | FHIR R4 (hand-built, no external SDK), FastAPI |
| Visualization | py3Dmol (3D protein-ligand), Plotly (interactive charts) |
| Frontend | Streamlit |
| CI | GitHub Actions (pytest, Python 3.10/3.11 matrix) |
| API hosting | Render |

---

## Related projects

- [ProteinSynergyDock](https://github.com/Aprameya05/ProteinSynergyDock) -- training code, model weights, and dataset preprocessing
- [ProteinWhisper](https://github.com/Aprameya05/ProteinWhisper) -- protein function encoder for zero-shot GO annotation
- [DrugSynergy3D](https://github.com/Aprameya05/DrugSynergy3D) -- SE(3)-equivariant synergy prediction using 3D molecular geometry
