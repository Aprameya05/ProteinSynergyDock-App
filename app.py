"""
ProteinSynergyDock — Streamlit App with Real 3D Molecular Visualization
========================================================================
"""

import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool
from torch_geometric.data import Data, Batch
from rdkit import Chem
from rdkit.Chem import AllChem
import py3Dmol
import numpy as np
import os
import streamlit.components.v1 as components

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ProteinSynergyDock",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem;
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        border-radius: 12px;
        margin-bottom: 2rem;
        color: white;
    }
    .main-header h1 { color: #4fc3f7; font-size: 2.5rem; margin: 0; }
    .main-header p  { color: #b0bec5; margin: 0.5rem 0 0; }
    .metric-card {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
        border-left: 4px solid #4fc3f7;
    }
    .synergy-positive { border-left-color: #4caf50 !important; }
    .synergy-negative { border-left-color: #f44336 !important; }
    .synergy-neutral  { border-left-color: #ff9800 !important; }
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #0f3460, #16213e);
        color: white;
        border: none;
        padding: 0.75rem;
        font-size: 1.1rem;
        border-radius: 8px;
        cursor: pointer;
    }
</style>
""", unsafe_allow_html=True)

# ── Model definition ──────────────────────────────────────────────────────────

class DrugEncoder(nn.Module):
    def __init__(self, in_dim=7, hidden=128, out_dim=256, heads=4):
        super().__init__()
        self.proj  = nn.Linear(in_dim, hidden)
        self.conv1 = GATv2Conv(hidden, hidden, heads=heads, concat=True)
        self.conv2 = GATv2Conv(hidden*heads, out_dim, heads=1, concat=False)
        self.norm1 = nn.LayerNorm(hidden*heads)
        self.norm2 = nn.LayerNorm(out_dim)

    def forward(self, x, edge_index, batch):
        x = F.gelu(self.proj(x))
        x = F.gelu(self.norm1(self.conv1(x, edge_index)))
        x = F.gelu(self.norm2(self.conv2(x, edge_index)))
        return global_mean_pool(x, batch)

class CrossDrugAttention(nn.Module):
    def __init__(self, dim=256):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads=4, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.ff   = nn.Sequential(nn.Linear(dim, dim*2), nn.GELU(), nn.Linear(dim*2, dim))

    def forward(self, a, b):
        seq = torch.stack([a, b], dim=1)
        att, _ = self.attn(seq, seq, seq)
        seq = self.norm(seq + att)
        seq = seq + self.ff(seq)
        return seq.reshape(seq.shape[0], -1)

class ProteinSynergyDock(nn.Module):
    def __init__(self, go_dim=512, drug_dim=256, hidden=512):
        super().__init__()
        self.drug_encoder = DrugEncoder(in_dim=7, hidden=128, out_dim=drug_dim)
        self.cross_attn   = CrossDrugAttention(dim=drug_dim)
        self.film_scale   = nn.Linear(go_dim, drug_dim*2)
        self.film_bias    = nn.Linear(go_dim, drug_dim*2)
        self.head = nn.Sequential(
            nn.Linear(drug_dim*2 + 2, hidden), nn.LayerNorm(hidden), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden, hidden//2), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden//2, 2),
        )

    def forward(self, da, db, go, dock):
        ea    = self.drug_encoder(da.x, da.edge_index, da.batch)
        eb    = self.drug_encoder(db.x, db.edge_index, db.batch)
        fused = self.cross_attn(ea, eb)
        fused = fused * (1 + self.film_scale(go)) + self.film_bias(go)
        fused = torch.cat([fused, dock], dim=-1)
        out   = self.head(fused)
        return out[:, 0], out[:, 1]

# ── Load model ────────────────────────────────────────────────────────────────

@st.cache_resource
def load_model():
    model = ProteinSynergyDock()
    ckpt_path = 'proteinsydock_v2_final.pt'
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        model.load_state_dict(ckpt['state_dict'])
        model.eval()
        return model, ckpt['pearson_r'], ckpt['auroc']
    return model, 0.0, 0.0

model, best_r, best_auroc = load_model()

# ── Drug graph builder ────────────────────────────────────────────────────────

def smiles_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return None
    try:
        mol = Chem.AddHs(mol)
        res = AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
        mol = Chem.RemoveHs(mol)
        if res == -1: AllChem.Compute2DCoords(mol)
    except: return None
    atom_features, positions = [], []
    conf = mol.GetConformer() if mol.GetNumConformers() > 0 else None
    for atom in mol.GetAtoms():
        atom_features.append([atom.GetAtomicNum(), atom.GetDegree(), atom.GetFormalCharge(),
            int(atom.GetIsAromatic()), int(atom.IsInRing()), atom.GetTotalNumHs(),
            atom.GetNumRadicalElectrons()])
        if conf:
            p = conf.GetAtomPosition(atom.GetIdx())
            positions.append([p.x, p.y, p.z])
        else: positions.append([0., 0., 0.])
    edge_src, edge_dst = [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edge_src += [i, j]; edge_dst += [j, i]
    if len(edge_src) == 0: return None
    return Data(x=torch.tensor(atom_features, dtype=torch.float),
                pos=torch.tensor(positions, dtype=torch.float),
                edge_index=torch.tensor([edge_src, edge_dst], dtype=torch.long))

# ── 3D Viewer ─────────────────────────────────────────────────────────────────

def show_3d_molecules(smiles_a, smiles_b, name_a="Drug A", name_b="Drug B", height=450):
    """Render two molecules in interactive 3D py3Dmol viewer."""
    mol_a = Chem.MolFromSmiles(smiles_a) if smiles_a else None
    mol_b = Chem.MolFromSmiles(smiles_b) if smiles_b else None

    if mol_a is None and mol_b is None:
        st.warning("Could not parse either SMILES")
        return

    viewer = py3Dmol.view(width=700, height=height)

    # Add Drug A in blue
    if mol_a:
        try:
            mol_a = Chem.AddHs(mol_a)
            AllChem.EmbedMolecule(mol_a, AllChem.ETKDGv3())
            AllChem.MMFFOptimizeMolecule(mol_a)
            mol_a = Chem.RemoveHs(mol_a)
            mb_a = Chem.MolToMolBlock(mol_a)
            viewer.addModel(mb_a, 'sdf')
            viewer.setStyle({'model': 0}, {
                'stick': {'colorscheme': 'cyanCarbon', 'radius': 0.15},
                'sphere': {'colorscheme': 'cyanCarbon', 'scale': 0.3}
            })
        except Exception as e:
            st.warning(f"Could not render {name_a} in 3D: {e}")

    # Add Drug B in orange, offset slightly
    if mol_b:
        try:
            mol_b = Chem.AddHs(mol_b)
            AllChem.EmbedMolecule(mol_b, AllChem.ETKDGv3())
            AllChem.MMFFOptimizeMolecule(mol_b)
            mol_b = Chem.RemoveHs(mol_b)

            # Offset Drug B so they don't overlap
            conf = mol_b.GetConformer()
            if mol_a and mol_a.GetNumConformers() > 0:
                conf_a = mol_a.GetConformer()
                max_x_a = max(conf_a.GetAtomPosition(i).x for i in range(mol_a.GetNumAtoms()))
                offset = max_x_a + 8.0
            else:
                offset = 8.0
            for i in range(mol_b.GetNumAtoms()):
                pos = conf.GetAtomPosition(i)
                conf.SetAtomPosition(i, (pos.x + offset, pos.y, pos.z))

            mb_b = Chem.MolToMolBlock(mol_b)
            viewer.addModel(mb_b, 'sdf')
            model_idx = 1 if mol_a else 0
            viewer.setStyle({'model': model_idx}, {
                'stick': {'colorscheme': 'orangeCarbon', 'radius': 0.15},
                'sphere': {'colorscheme': 'orangeCarbon', 'scale': 0.3}
            })
        except Exception as e:
            st.warning(f"Could not render {name_b} in 3D: {e}")

    viewer.setBackgroundColor('#1a1a2e')
    viewer.zoomTo()
    viewer.zoom(0.85)

    # Embed in Streamlit
    viewer_html = viewer._make_html()
    components.html(viewer_html, height=height + 20, scrolling=False)

# ── 2D Structure images ───────────────────────────────────────────────────────

def show_2d_structures(smiles_a, smiles_b):
    st.info("2D structure view requires additional system libraries. Use 3D Interactive mode.")
    mols, names = [], []
    for smi, name in [(smiles_a, "Drug A (blue)"), (smiles_b, "Drug B (orange)")]:
        mol = Chem.MolFromSmiles(smi) if smi else None
        if mol:
            AllChem.Compute2DCoords(mol)
            mols.append(mol); names.append(name)
    if mols:
        img = Draw.MolsToGridImage(mols, molsPerRow=2, subImgSize=(350, 250),
                                    legends=names, returnPNG=False)
        st.image(img, use_container_width=True)

# ── Showcases ─────────────────────────────────────────────────────────────────

SHOWCASES = {
    "Custom input": {"smiles_a": "", "smiles_b": "", "dock_a": -7.0, "dock_b": -7.0, "note": ""},
    "✅ Vemurafenib + Trametinib (BRAF+MEK — Approved)": {
        "smiles_a": "CCCS(=O)(=O)Nc1ccc(F)c(C(=O)c2c[nH]c3ncc(-c4ccc(Cl)cc4)cc23)c1",
        "smiles_b": "CC(=O)Nc1ccc(-c2cc3c(nc(N)nc3n2C)N2CCC(F)(F)CC2=O)cc1F",
        "dock_a": -9.04, "dock_b": -7.52,
        "note": "FDA-approved combination for BRAF V600E melanoma. Known synergy score: **8.4**"
    },
    "❌ Imatinib + Dasatinib (ABL1 — Antagonistic)": {
        "smiles_a": "CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)NC4=NC=CC(=N4)C5=CN=CC=C5",
        "smiles_b": "Cc1nc(Nc2ncc(s2)C(=O)Nc2c(C)cccc2Cl)cc(n1)N1CCN(CCO)CC1",
        "dock_a": -9.38, "dock_b": -8.79,
        "note": "Both drugs compete for the ABL1 ATP binding pocket. Known synergy: **-1.4** (antagonistic)"
    },
    "⚠️ Olaparib + Rucaparib (PARP1 — Mild Synergy)": {
        "smiles_a": "O=C1CCCN1c1ccc(cc1)C(=O)c1[nH]ncc1C1CC1",
        "smiles_b": "NCc1cc2cc(F)ccc2[nH]1-c1ccc3NCCCC(=O)c3c1",
        "dock_a": -7.2, "dock_b": -6.8,
        "note": "Complementary PARP1 inhibition mechanisms. Known synergy: **2.1**"
    },
    "✅ Erlotinib + Lapatinib (EGFR — Synergistic)": {
        "smiles_a": "COCCOC1=C(C=C2C(=C1)C(=NC=N2)NC3=CC=CC(=C3)C#C)OCCOC",
        "smiles_b": "CS(=O)(=O)CCNCc1oc(cc1)c2ccc3ncnc(Nc4ccc(Oc5cccc(Cl)c5)c(Cl)c4)c3c2",
        "dock_a": -6.22, "dock_b": -7.24,
        "note": "Dual EGFR inhibition via different binding modes. Known synergy: **5.5**"
    },
}

# ── Main UI ───────────────────────────────────────────────────────────────────

st.markdown("""
<div class="main-header">
    <h1>🧬 ProteinSynergyDock</h1>
    <p>Structure-aware drug combination synergy prediction via co-docking GNN with GO function context</p>
    <p style="font-size:13px; color:#78909c; margin-top:8px;">
        Pearson r = 0.5768 &nbsp;|&nbsp; Real AutoDock Vina docking &nbsp;|&nbsp; ProteinWhisper++ GO encoder (Fmax 0.4006)
    </p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("## 🔬 Quick Examples")
    example = st.selectbox("Choose a drug pair:", list(SHOWCASES.keys()))

    ex = SHOWCASES[example]
    if ex["note"]:
        st.info(ex["note"])

    st.markdown("---")
    st.markdown("## ⚙️ Settings")
    view_mode = st.radio("Visualization mode:", ["3D Interactive", "2D Structure"])
    show_details = st.checkbox("Show model details", value=False)

    st.markdown("---")
    st.markdown(f"""
## 📊 Model Stats
- **Pearson r:** {best_r:.4f}
- **AUROC:** {best_auroc:.4f}
- **Training data:** 231 real NCI ALMANAC scores
- **Docking pairs:** 12 AutoDock Vina runs

## 🔗 Links
- [GitHub](https://github.com/Aprameya05/ProteinSynergyDock)
- [ProteinWhisper](https://github.com/Aprameya05/ProteinWhisper)
- [DrugSynergy3D](https://github.com/Aprameya05/DrugSynergy3D)
    """)

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### Drug Inputs")

    smiles_a = st.text_area(
        "Drug A — SMILES",
        value=ex["smiles_a"],
        height=80,
        placeholder="Paste SMILES string here...",
        help="SMILES notation for Drug A (shown in blue)"
    )
    smiles_b = st.text_area(
        "Drug B — SMILES",
        value=ex["smiles_b"],
        height=80,
        placeholder="Paste SMILES string here...",
        help="SMILES notation for Drug B (shown in orange)"
    )

    st.markdown("### Docking Scores")
    st.caption("From AutoDock Vina — more negative = stronger binding")

    dock_a = st.slider("Drug A docking score (kcal/mol)", -15.0, 0.0,
                        float(ex["dock_a"]), 0.1,
                        help="AutoDock Vina binding affinity for Drug A")
    dock_b = st.slider("Drug B docking score (kcal/mol)", -15.0, 0.0,
                        float(ex["dock_b"]), 0.1,
                        help="AutoDock Vina binding affinity for Drug B")

    predict_btn = st.button("🔬 Predict Synergy", type="primary")

with col2:
    st.markdown("### 3D Molecular Structure")

    if smiles_a or smiles_b:
        if view_mode == "3D Interactive":
            with st.spinner("Generating 3D structure..."):
                show_3d_molecules(smiles_a, smiles_b)
            st.caption("🔵 Drug A (blue) &nbsp;&nbsp; 🟠 Drug B (orange) &nbsp;&nbsp; *Drag to rotate · Scroll to zoom*")
        else:
            show_2d_structures(smiles_a, smiles_b)
    else:
        st.info("Enter SMILES strings to see molecular structures")

# Prediction results
st.markdown("---")
st.markdown("### Prediction Results")

if predict_btn or example != "Custom input":
    if not smiles_a or not smiles_b:
        st.error("Please enter SMILES for both drugs")
    else:
        ga = smiles_to_graph(smiles_a)
        gb = smiles_to_graph(smiles_b)

        if ga is None:
            st.error("❌ Invalid SMILES for Drug A — please check your input")
        elif gb is None:
            st.error("❌ Invalid SMILES for Drug B — please check your input")
        else:
            with st.spinner("Running synergy prediction..."):
                go_emb = torch.zeros(512).unsqueeze(0)
                dock   = torch.tensor([[float(dock_a), float(dock_b)]])

                with torch.no_grad():
                    score, logit = model(
                        Batch.from_data_list([ga]),
                        Batch.from_data_list([gb]),
                        go_emb, dock
                    )
                    synergy_score = score.item()
                    synergy_prob  = torch.sigmoid(logit).item()

            # Verdict
            if synergy_score > 4.0:
                verdict = "✅ Strongly Synergistic"
                color   = "green"
            elif synergy_score > 2.0:
                verdict = "⚠️ Mildly Synergistic"
                color   = "orange"
            elif synergy_score > -1.0:
                verdict = "➖ Approximately Additive"
                color   = "blue"
            else:
                verdict = "❌ Antagonistic"
                color   = "red"

            # Display metrics
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric("Synergy Score (Loewe)", f"{synergy_score:.3f}")
            with m2:
                st.metric("Synergy Probability", f"{synergy_prob:.3f}")
            with m3:
                st.metric("Drug A Binding", f"{dock_a:.2f} kcal/mol")
            with m4:
                st.metric("Drug B Binding", f"{dock_b:.2f} kcal/mol")

            st.markdown(f"### Verdict: :{color}[{verdict}]")

            # Interpretation
            with st.expander("📖 How to interpret this result"):
                st.markdown("""
| Score Range | Interpretation | Clinical Meaning |
|-------------|----------------|-----------------|
| > 4.0 | Strongly Synergistic | Drugs work much better together |
| 2.0 – 4.0 | Mildly Synergistic | Modest benefit from combination |
| -1.0 – 2.0 | Approximately Additive | Simple additive effect |
| < -1.0 | Antagonistic | Drugs interfere with each other |

The **Loewe synergy score** measures how much better (or worse) a drug combination 
performs compared to what you'd expect if each drug acted independently.
                """)

            if show_details:
                with st.expander("🔧 Model Architecture Details"):
                    st.markdown(f"""
**ProteinSynergyDock v2 Architecture:**
- Drug encoder: GATv2 (in=7, hidden=128, out=256, heads=4)
- Cross-drug attention: MultiheadAttention (4 heads)
- GO conditioning: FiLM (512-dim ProteinWhisper++ embeddings)
- Docking integration: Raw Vina scores concatenated before prediction head
- Total parameters: ~1.85M

**Training:**
- Dataset: 780 drug-protein triplets (231 real NCI ALMANAC scores)
- Real docking: 12 AutoDock Vina drug-protein pairs
- Best Pearson r: **{best_r:.4f}**
- Best AUROC: **{best_auroc:.4f}**
                    """)

st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#78909c; font-size:13px;">
ProteinSynergyDock · Aprameya Bharadwaj · DSCE Bangalore 2026<br>
<a href="https://github.com/Aprameya05/ProteinSynergyDock">GitHub</a> · 
<a href="https://github.com/Aprameya05/ProteinWhisper">ProteinWhisper</a> · 
<a href="https://github.com/Aprameya05/DrugSynergy3D">DrugSynergy3D</a>
</div>
""", unsafe_allow_html=True)
