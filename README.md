# ProteinSynergyDock — Interactive 3D Molecular Docking App

**Live demo:** [Open App](https://aprameya05-proteinsynergydock-app.streamlit.app)

---

## What this app does

Most drug synergy prediction tools take two drug SMILES strings and output a score based purely on chemical structure. They never ask: *where does the drug physically sit inside the protein?* or *what does the protein actually do biologically?*

This app is the first tool to combine all three:
- **3D docking geometry** — where each drug binds in the protein pocket
- **Protein biological function** — what the protein does (GO terms)
- **Drug pair chemistry** — molecular graph of both drugs

---

## How it works — step by step

### What the user inputs
1. Drug A name + SMILES string
2. Drug B name + SMILES string
3. PDB ID of the target protein (from rcsb.org)

### What the app does automatically

**Step 1 — Fetch protein structure**
Downloads the experimental 3D crystal structure of the target protein directly from RCSB PDB in real time. No manual upload needed.

**Step 2 — Prepare ligands**
Converts both drug SMILES strings into 3D atomic coordinates using RDKit (ETKDGv3 conformer generation + MMFF optimization).

**Step 3 — Run AutoDock Vina**
Performs real molecular docking of each drug against the protein binding pocket. Vina searches through thousands of possible drug orientations and finds the lowest-energy (best) binding pose. Outputs a binding affinity score in kcal/mol (more negative = stronger binding).

**Step 4 — Predict protein function**
ProteinWhisper++ reads the protein sequence and predicts its Gene Ontology (GO) function terms — biological process, molecular function, cellular component. This gives the model functional context about what the protein does, not just its structure.

**Step 5 — Predict synergy**
ProteinSynergyDock takes all of the above and runs it through a GNN:
- Drug A molecular graph (GATv2 encoder)
- Drug B molecular graph (GATv2 encoder)
- Cross-drug attention layer (models geometric complementarity between the two drugs)
- FiLM conditioning with GO protein function embeddings
- Real docking scores concatenated to the prediction head
- Output: Loewe synergy score + synergy probability

**Step 6 — Visualize in 3D**
Renders both drug molecules in an interactive py3Dmol viewer. Drag to rotate, scroll to zoom.

---

## Interpreting the output

| Synergy Score | Meaning | Clinical Implication |
|---------------|---------|---------------------|
| > 4.0 | Strongly Synergistic | Strong candidate for combination therapy |
| 2.0 – 4.0 | Mildly Synergistic | Modest benefit from combining |
| -1.0 – 2.0 | Approximately Additive | Drugs act independently |
| < -1.0 | Antagonistic | Drugs interfere — avoid combination |

The **Loewe synergy score** measures how much better (or worse) a drug combination performs compared to what you would expect if each drug acted independently at the same doses.

---

## Model performance

| Metric | Value |
|--------|-------|
| Pearson r | 0.5768 |
| AUROC | 0.5408 |
| Training data | 231 real NCI ALMANAC synergy scores |
| Real docking pairs | 12 AutoDock Vina runs |
| ProteinWhisper++ Fmax | 0.4006 (7.9x over baseline) |

---

## Architecture
Drug A (SMILES)

↓

RDKit 3D          ┐

GATv2 encoder     ├──→ Cross-drug attention ──→ FiLM (GO context) ──→ Synergy head

GATv2 encoder     ┘              ↑                     ↑

Drug B (SMILES)            AutoDock Vina          ProteinWhisper++

↓                    binding scores         GO embeddings

RDKit 3D                       ↑                     ↑

Vina docking          Protein sequence

(auto-run)            → 38,245 GO terms
---

## Tech stack

- **Molecular docking:** AutoDock Vina + OpenBabel
- **Drug encoding:** RDKit + PyTorch Geometric GATv2
- **Protein function:** ProteinWhisper++ (ESM-2 650M + GO DAG decoder)
- **Synergy model:** Cross-drug attention GNN + FiLM conditioning
- **Visualization:** py3Dmol (interactive 3D)
- **Frontend:** Streamlit

---

## Example drug pairs to try

| Drug A | Drug B | PDB ID | Expected |
|--------|--------|--------|----------|
| Vemurafenib | Trametinib | 3OG7 | ✅ Strongly synergistic (BRAF+MEK, FDA approved) |
| Imatinib | Dasatinib | 2HYY | ❌ Antagonistic (both compete for ABL1 ATP pocket) |
| Erlotinib | Lapatinib | 1IVO | ✅ Synergistic (dual EGFR inhibition) |
| Olaparib | Rucaparib | 4DQY | ⚠️ Mildly synergistic (complementary PARP1 inhibition) |

SMILES for all examples are pre-loaded in the Quick Examples dropdown.

---

## Related repositories

- [ProteinSynergyDock](https://github.com/Aprameya05/ProteinSynergyDock) — full model training code, data pipeline, checkpoints
- [ProteinWhisper](https://github.com/Aprameya05/ProteinWhisper) — zero-shot protein function annotation for the dark proteome
- [DrugSynergy3D](https://github.com/Aprameya05/DrugSynergy3D) — SE(3) equivariant GNN for drug combination synergy
