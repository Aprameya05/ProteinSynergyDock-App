# ProteinSynergyDock — Interactive Demo

**Live app →** [proteinsydock.streamlit.app](https://aprameya05-proteinsynergydock-app.streamlit.app)

---

![Docking visualization](demo.png.png)
*Vemurafenib (cyan) and Trametinib (orange) docked inside BRAF kinase. Both drugs automatically positioned in the binding pocket via AutoDock Vina.*

---

## What this does

Input two drugs + a protein + a cancer cell line → the app automatically:

1. Fetches the protein 3D structure from RCSB PDB
2. Docks both drugs into the binding pocket using AutoDock Vina (real docking, no shortcuts)
3. Detects the binding pocket from co-crystallized ligand coordinates (HETATM method)
4. Renders both docked drugs inside the protein ribbon in interactive 3D
5. Predicts synergy score specific to the selected cancer cell line
6. Compares prediction against NCI ALMANAC ground truth if the pair is known

## Features

**Drug inputs**
- Select from 35+ known cancer drugs with SMILES auto-filled
- Or paste any custom SMILES string

**Cancer context**
- 9 cancer panels (Melanoma, NSCLC, Breast, Leukemia, Ovarian, CNS, Colon, Renal, Prostate)
- 60 NCI-60 cell lines with cell-line-specific predictions

**3D visualization**
- Interactive py3Dmol viewer — drag to rotate, scroll to zoom
- Protein shown as spectrum-colored ribbon
- Drug A in cyan, Drug B in orange
- Binding pocket detected from crystal structure ligand coordinates

**Ground truth comparison**
- Looks up NCI ALMANAC known synergy score if the drug pair exists in the database
- Shows prediction error vs ground truth

**Results history**
- Last 5 predictions saved in sidebar for comparison

## Example drug pairs

| Drug A | Drug B | PDB | Cancer | Known synergy |
|--------|--------|-----|--------|---------------|
| Vemurafenib | Trametinib | 3OG7 | Melanoma / UACC-62 | 8.4 ✅ |
| Imatinib | Dasatinib | 2HYY | Leukemia / K-562 | -1.4 ❌ |
| Erlotinib | Lapatinib | 1IVO | NSCLC / A549 | 5.5 ✅ |
| Olaparib | Rucaparib | 4DQY | Ovarian / OVCAR-3 | 2.1 ⚠️ |

All pre-loaded in the Quick Examples dropdown.

## Model

- **Architecture:** GATv2 drug encoder + cross-drug attention + FiLM GO conditioning + cell line embedding
- **Pearson r:** 0.5768 | **AUROC:** 0.5408
- **Training:** 107,103 real NCI ALMANAC synergy measurements
- **Docking:** 842 AutoDock Vina runs across 20 cancer target proteins
- **Protein function:** ProteinWhisper++ (Fmax 0.4006, 7.9× over baseline)

## Related

- [ProteinSynergyDock](https://github.com/Aprameya05/ProteinSynergyDock) — model training code and weights
- [ProteinWhisper](https://github.com/Aprameya05/ProteinWhisper) — protein function encoder
- [DrugSynergy3D](https://github.com/Aprameya05/DrugSynergy3D) — SE(3) equivariant synergy prediction
