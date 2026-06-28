"""
ProteinSynergyDock v2 — Full Auto-Docking Pipeline
===================================================
User inputs: Drug A SMILES + Drug B SMILES + PDB ID
App runs: AutoDock Vina automatically
Shows: Both drugs docked IN the protein + synergy prediction
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
import requests
import subprocess
import tempfile
import time
import streamlit.components.v1 as components

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ProteinSynergyDock",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        text-align: center; padding: 2rem;
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        border-radius: 12px; margin-bottom: 2rem; color: white;
    }
    .main-header h1 { color: #4fc3f7; font-size: 2.5rem; margin: 0; }
    .main-header p  { color: #b0bec5; margin: 0.5rem 0 0; }
    .step-box {
        background: #f8f9fa; border-radius: 8px; padding: 1rem;
        border-left: 4px solid #4fc3f7; margin: 0.5rem 0;
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

# ── Install Vina if needed ────────────────────────────────────────────────────

@st.cache_resource
def ensure_vina():
    try:
        from vina import Vina
        return True
    except ImportError:
        subprocess.run(['pip', 'install', 'vina', '-q'], capture_output=True)
        try:
            from vina import Vina
            return True
        except:
            return False

@st.cache_resource
def ensure_obabel():
    result = subprocess.run(['which', 'obabel'], capture_output=True, text=True)
    if result.returncode == 0:
        return True
    subprocess.run(['apt-get', 'install', '-y', '-q', 'openbabel'], capture_output=True)
    result = subprocess.run(['which', 'obabel'], capture_output=True, text=True)
    return result.returncode == 0

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

# ── PDB fetcher ───────────────────────────────────────────────────────────────

def fetch_pdb(pdb_id, save_dir):
    """Download PDB structure from RCSB."""
    pdb_path = os.path.join(save_dir, f"{pdb_id}.pdb")
    if os.path.exists(pdb_path) and os.path.getsize(pdb_path) > 1000:
        return pdb_path
    url = f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb"
    r = requests.get(url, timeout=30)
    if r.status_code == 200:
        with open(pdb_path, 'w') as f:
            f.write(r.text)
        return pdb_path
    return None

def get_protein_info(pdb_id):
    """Get protein name and organism from RCSB API."""
    try:
        url = f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id.upper()}"
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            data = r.json()
            title = data.get('struct', {}).get('title', 'Unknown protein')
            return title
    except:
        pass
    return f"Protein {pdb_id}"

# ── AutoDock Vina docking ─────────────────────────────────────────────────────

def get_binding_box(pdb_path, padding=10.0):
    """Auto-detect binding box from protein geometric center."""
    coords = []
    with open(pdb_path) as f:
        for line in f:
            if line.startswith(('ATOM', 'HETATM')):
                try:
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    coords.append([x, y, z])
                except:
                    pass
    if not coords:
        return [0, 0, 0], [30, 30, 30]
    coords = np.array(coords)
    center = coords.mean(axis=0).tolist()
    size   = np.clip(coords.max(axis=0) - coords.min(axis=0) + padding, 20, 30).tolist()
    return center, size

def prepare_ligand_pdbqt(smiles, name, work_dir):
    """Convert SMILES to PDBQT for Vina."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return None
    try:
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
        AllChem.MMFFOptimizeMolecule(mol)
        mol = Chem.RemoveHs(mol)
    except:
        try: AllChem.Compute2DCoords(mol)
        except: return None

    sdf_path   = os.path.join(work_dir, f"{name}.sdf")
    pdb_path   = os.path.join(work_dir, f"{name}.pdb")
    pdbqt_path = os.path.join(work_dir, f"{name}.pdbqt")

    writer = Chem.SDWriter(sdf_path)
    writer.write(mol)
    writer.close()

    subprocess.run(['obabel', sdf_path, '-O', pdb_path, '-h'], capture_output=True)
    subprocess.run(['obabel', pdb_path, '-O', pdbqt_path, '--partialcharge', 'gasteiger'],
                   capture_output=True)

    if os.path.exists(pdbqt_path) and os.path.getsize(pdbqt_path) > 0:
        return pdbqt_path
    return None

def prepare_receptor_pdbqt(pdb_path, work_dir):
    """Convert PDB receptor to PDBQT."""
    pdb_id     = os.path.basename(pdb_path).replace('.pdb', '')
    pdbqt_path = os.path.join(work_dir, f"{pdb_id}_receptor.pdbqt")
    if os.path.exists(pdbqt_path) and os.path.getsize(pdbqt_path) > 0:
        return pdbqt_path
    subprocess.run(['obabel', pdb_path, '-O', pdbqt_path,
                    '--partialcharge', 'gasteiger'], capture_output=True)
    if os.path.exists(pdbqt_path) and os.path.getsize(pdbqt_path) > 0:
        return pdbqt_path
    return None

def run_vina_docking(receptor_pdbqt, ligand_pdbqt, center, size, out_path, exhaustiveness=8):
    """Run AutoDock Vina binary via subprocess."""
    try:
        vina_cmd = None
        for cmd in ['vina', 'autodock_vina', '/usr/bin/vina']:
            r = subprocess.run([cmd, '--version'], capture_output=True)
            if r.returncode == 0:
                vina_cmd = cmd
                break
        if vina_cmd is None:
            return None, "vina not found"
        result = subprocess.run([
            vina_cmd,
            '--receptor', receptor_pdbqt,
            '--ligand', ligand_pdbqt,
            '--out', out_path,
            '--center_x', str(round(center[0],3)),
            '--center_y', str(round(center[1],3)),
            '--center_z', str(round(center[2],3)),
            '--size_x', str(round(size[0],3)),
            '--size_y', str(round(size[1],3)),
            '--size_z', str(round(size[2],3)),
            '--exhaustiveness', str(exhaustiveness),
            '--num_modes', '3',
        ], capture_output=True, text=True, timeout=300)
        best_score = None
        for line in result.stdout.split('\n'):
            if line.strip().startswith('1 '):
                try: best_score = float(line.split()[1]); break
                except: pass
        if best_score is None and os.path.exists(out_path):
            with open(out_path) as f:
                for line in f:
                    if 'REMARK VINA RESULT' in line:
                        try: best_score = float(line.split()[3]); break
                        except: pass
        return best_score, out_path
    except Exception as e:
         return None, str(e)

def read_pdbqt_molblock(pdbqt_path):
    """Read first pose from PDBQT file as atom coords."""
    atoms = []
    if not os.path.exists(pdbqt_path):
        return None
    with open(pdbqt_path) as f:
        for line in f:
            if line.startswith('ENDMDL'):
                break
            if line.startswith(('ATOM', 'HETATM')):
                try:
                    atom = line[12:16].strip()
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    atoms.append((atom, x, y, z))
                except:
                    pass
    return atoms

# ── 3D Viewer with protein + both docked drugs ────────────────────────────────

def show_docking_result(pdb_content, pose_atoms_a, pose_atoms_b, pdb_id):
    """Show protein + both docked drugs in py3Dmol."""
    viewer = py3Dmol.view(width=750, height=500)

    # Add protein
    viewer.addModel(pdb_content, 'pdb')
    viewer.setStyle({'model': 0}, {
        'cartoon': {'color': 'spectrum', 'opacity': 0.7},
    })
    viewer.addSurface(py3Dmol.SAS, {'opacity': 0.1, 'color': 'white'}, {'model': 0})

    # Add Drug A pose (cyan sticks)
    if pose_atoms_a:
        xyz_block = "MODEL 1\n"
        for i, (atom, x, y, z) in enumerate(pose_atoms_a):
            xyz_block += f"HETATM{i+1:5d}  {atom:<4s}LGA A   1    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          C\n"
        xyz_block += "ENDMDL\n"
        viewer.addModel(xyz_block, 'pdb')
        viewer.setStyle({'model': 1}, {
            'stick': {'colorscheme': 'cyanCarbon', 'radius': 0.2},
            'sphere': {'colorscheme': 'cyanCarbon', 'scale': 0.35}
        })

    # Add Drug B pose (orange sticks)
    if pose_atoms_b:
        xyz_block = "MODEL 1\n"
        for i, (atom, x, y, z) in enumerate(pose_atoms_b):
            xyz_block += f"HETATM{i+1:5d}  {atom:<4s}LGB B   1    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          C\n"
        xyz_block += "ENDMDL\n"
        viewer.addModel(xyz_block, 'pdb')
        model_idx = 2 if pose_atoms_a else 1
        viewer.setStyle({'model': model_idx}, {
            'stick': {'colorscheme': 'orangeCarbon', 'radius': 0.2},
            'sphere': {'colorscheme': 'orangeCarbon', 'scale': 0.35}
        })

    viewer.setBackgroundColor('#1a1a2e')
    viewer.zoomTo({'model': 1} if pose_atoms_a else {})
    viewer.zoom(1.2)

    html = viewer._make_html()
    components.html(html, height=520, scrolling=False)

def show_drugs_only_3d(smiles_a, smiles_b):
    """Fallback: show just the drug molecules in 3D without protein."""
    viewer = py3Dmol.view(width=750, height=400)
    offset = 0

    for smiles, color, name in [(smiles_a, 'cyanCarbon', 'Drug A'), (smiles_b, 'orangeCarbon', 'Drug B')]:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None: continue
        try:
            mol = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
            AllChem.MMFFOptimizeMolecule(mol)
            mol = Chem.RemoveHs(mol)
            conf = mol.GetConformer()
            for i in range(mol.GetNumAtoms()):
                pos = conf.GetAtomPosition(i)
                conf.SetAtomPosition(i, (pos.x + offset, pos.y, pos.z))
            mb = Chem.MolToMolBlock(mol)
            viewer.addModel(mb, 'sdf')
            viewer.setStyle({'model': offset//15}, {
                'stick': {'colorscheme': color, 'radius': 0.15},
                'sphere': {'colorscheme': color, 'scale': 0.3}
            })
            offset += 15
        except: pass

    viewer.setBackgroundColor('#1a1a2e')
    viewer.zoomTo()
    components.html(viewer._make_html(), height=420, scrolling=False)

# ── Showcases ─────────────────────────────────────────────────────────────────

SHOWCASES = {
    "Custom input": {
        "smiles_a": "", "smiles_b": "", "pdb_id": "",
        "name_a": "", "name_b": "", "note": ""
    },
    "✅ Vemurafenib + Trametinib on BRAF (Approved Combo)": {
        "smiles_a": "CCCS(=O)(=O)Nc1ccc(F)c(C(=O)c2c[nH]c3ncc(-c4ccc(Cl)cc4)cc23)c1",
        "smiles_b": "CC(=O)Nc1ccc(-c2cc3c(nc(N)nc3n2C)N2CCC(F)(F)CC2=O)cc1F",
        "pdb_id": "3OG7",
        "name_a": "Vemurafenib", "name_b": "Trametinib",
        "note": "FDA-approved BRAF+MEK combination for melanoma. Known synergy: **8.4**"
    },
    "❌ Imatinib + Dasatinib on ABL1 (Antagonistic)": {
        "smiles_a": "CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)NC4=NC=CC(=N4)C5=CN=CC=C5",
        "smiles_b": "Cc1nc(Nc2ncc(s2)C(=O)Nc2c(C)cccc2Cl)cc(n1)N1CCN(CCO)CC1",
        "pdb_id": "2HYY",
        "name_a": "Imatinib", "name_b": "Dasatinib",
        "note": "Both compete for ABL1 ATP pocket — redundant. Known synergy: **-1.4**"
    },
    "✅ Erlotinib + Lapatinib on EGFR (Synergistic)": {
        "smiles_a": "COCCOC1=C(C=C2C(=C1)C(=NC=N2)NC3=CC=CC(=C3)C#C)OCCOC",
        "smiles_b": "CS(=O)(=O)CCNCc1oc(cc1)c2ccc3ncnc(Nc4ccc(Oc5cccc(Cl)c5)c(Cl)c4)c3c2",
        "pdb_id": "1IVO",
        "name_a": "Erlotinib", "name_b": "Lapatinib",
        "note": "Dual EGFR inhibition. Known synergy: **5.5**"
    },
    "⚠️ Olaparib + Rucaparib on PARP1 (Mild Synergy)": {
        "smiles_a": "O=C1CCCN1c1ccc(cc1)C(=O)c1[nH]ncc1C1CC1",
        "smiles_b": "NCc1cc2cc(F)ccc2[nH]1-c1ccc3NCCCC(=O)c3c1",
        "pdb_id": "4DQY",
        "name_a": "Olaparib", "name_b": "Rucaparib",
        "note": "Complementary PARP1 inhibition. Known synergy: **2.1**"
    },
}

# ── Main UI ───────────────────────────────────────────────────────────────────

st.markdown("""
<div class="main-header">
    <h1>🧬 ProteinSynergyDock</h1>
    <p>Automatic molecular docking + drug combination synergy prediction</p>
    <p style="font-size:13px; color:#78909c; margin-top:8px;">
        Input two drugs + a protein → auto-docking → 3D visualization → synergy score
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
    st.markdown(f"""
## 📊 Model Stats
- **Pearson r:** {best_r:.4f}
- **AUROC:** {best_auroc:.4f}
- **Real docking:** AutoDock Vina
- **Data:** 231 NCI ALMANAC scores

## 🔗 Links
- [GitHub](https://github.com/Aprameya05/ProteinSynergyDock)
- [ProteinWhisper](https://github.com/Aprameya05/ProteinWhisper)
- [DrugSynergy3D](https://github.com/Aprameya05/DrugSynergy3D)
    """)

# Main layout
col1, col2 = st.columns([1, 1.2])

with col1:
    st.markdown("### 💊 Drug Inputs")

    name_a = st.text_input("Drug A name (optional)", value=ex.get("name_a", ""), placeholder="e.g. Imatinib")
    smiles_a = st.text_area("Drug A — SMILES", value=ex["smiles_a"], height=80, placeholder="Paste SMILES...")

    name_b = st.text_input("Drug B name (optional)", value=ex.get("name_b", ""), placeholder="e.g. Dasatinib")
    smiles_b = st.text_area("Drug B — SMILES", value=ex["smiles_b"], height=80, placeholder="Paste SMILES...")

    st.markdown("### 🧫 Target Protein")
    pdb_id = st.text_input(
        "PDB ID (from rcsb.org)",
        value=ex.get("pdb_id", ""),
        placeholder="e.g. 2HYY, 1IVO, 3OG7",
        help="4-character PDB ID. Find yours at rcsb.org"
    ).strip().upper()

    if pdb_id:
        st.caption(f"Will fetch: https://files.rcsb.org/download/{pdb_id}.pdb")

    exhaustiveness = st.slider("Docking exhaustiveness (higher = more accurate, slower)",
                                4, 16, 8, 2)

    run_btn = st.button("🔬 Run Docking + Predict Synergy", type="primary")

with col2:
    st.markdown("### 🔭 3D Visualization")
    viz_placeholder = st.empty()

    if smiles_a or smiles_b:
        with viz_placeholder.container():
            st.caption("Preview (pre-docking)")
            show_drugs_only_3d(smiles_a, smiles_b)
            st.caption("🔵 Drug A &nbsp; 🟠 Drug B &nbsp; *Drag to rotate · Scroll to zoom*")

# ── Full pipeline on button click ─────────────────────────────────────────────

if run_btn:
    # Debug: check what's available
    import shutil
    vina_found = shutil.which('vina') or shutil.which('autodock_vina') or shutil.which('vina_1.2.6_linux_x86_64')
    obabel_found = shutil.which('obabel')
    st.info(f"Vina: {vina_found} | Obabel: {obabel_found}")
    if not smiles_a or not smiles_b:
        st.error("Please enter SMILES for both drugs")
        st.stop()
    if not pdb_id:
        st.error("Please enter a PDB ID")
        st.stop()

    ga = smiles_to_graph(smiles_a)
    gb = smiles_to_graph(smiles_b)
    if ga is None:
        st.error("❌ Invalid SMILES for Drug A")
        st.stop()
    if gb is None:
        st.error("❌ Invalid SMILES for Drug B")
        st.stop()

    st.markdown("---")
    st.markdown("### 🔄 Pipeline Running...")

    progress = st.progress(0)
    status   = st.status("Starting pipeline...", expanded=True)

    with tempfile.TemporaryDirectory() as work_dir:

        # Step 1: Fetch PDB
        with status:
            st.write(f"📥 Fetching {pdb_id} from RCSB PDB...")
        progress.progress(10)

        pdb_path = fetch_pdb(pdb_id, work_dir)
        if pdb_path is None:
            st.error(f"❌ Could not fetch PDB {pdb_id} — check the ID and try again")
            st.stop()

        protein_name = get_protein_info(pdb_id)
        with status:
            st.write(f"✅ Fetched: {protein_name}")
        progress.progress(20)

        pdb_content = open(pdb_path).read()

        # Step 2: Prepare receptor
        with status:
            st.write("⚙️ Preparing receptor (PDBQT conversion)...")
        ensure_obabel()
        receptor_pdbqt = prepare_receptor_pdbqt(pdb_path, work_dir)
        if receptor_pdbqt is None:
            st.warning("⚠️ Receptor prep failed — using structure-free prediction")
            receptor_pdbqt = None
        else:
            with status:
                st.write("✅ Receptor ready")
        progress.progress(30)

        # Step 3: Prepare ligands
        with status:
            st.write(f"💊 Preparing {name_a or 'Drug A'} ligand...")
        lig_a = prepare_ligand_pdbqt(smiles_a, "drug_a", work_dir)
        with status:
            st.write(f"💊 Preparing {name_b or 'Drug B'} ligand...")
        lig_b = prepare_ligand_pdbqt(smiles_b, "drug_b", work_dir)
        progress.progress(40)

        # Step 4: Docking
        center, size = get_binding_box(pdb_path)
        dock_score_a = -7.0
        dock_score_b = -7.0
        pose_atoms_a = None
        pose_atoms_b = None

        if receptor_pdbqt and lig_a:
            with status:
                st.write(f"🔬 Docking {name_a or 'Drug A'} → {pdb_id}...")
            ensure_vina()
            out_a = os.path.join(work_dir, "drug_a_docked.pdbqt")
            score_a, result_a = run_vina_docking(receptor_pdbqt, lig_a, center, size, out_a, exhaustiveness)
            if score_a is not None:
                dock_score_a = score_a
                pose_atoms_a = read_pdbqt_molblock(out_a)
                with status:
                    st.write(f"✅ {name_a or 'Drug A'}: {score_a:.2f} kcal/mol")
            else:
                with status:
                    st.write(f"⚠️ {name_a or 'Drug A'} docking failed — using default score")
        progress.progress(65)

        if receptor_pdbqt and lig_b:
            with status:
                st.write(f"🔬 Docking {name_b or 'Drug B'} → {pdb_id}...")
            out_b = os.path.join(work_dir, "drug_b_docked.pdbqt")
            score_b, result_b = run_vina_docking(receptor_pdbqt, lig_b, center, size, out_b, exhaustiveness)
            if score_b is not None:
                dock_score_b = score_b
                pose_atoms_b = read_pdbqt_molblock(out_b)
                with status:
                    st.write(f"✅ {name_b or 'Drug B'}: {score_b:.2f} kcal/mol")
            else:
                with status:
                    st.write(f"⚠️ {name_b or 'Drug B'} docking failed — using default score")
        progress.progress(80)

        # Step 5: Synergy prediction
        with status:
            st.write("🧠 Running synergy prediction...")

        go_emb = torch.zeros(512).unsqueeze(0)
        dock   = torch.tensor([[float(dock_score_a), float(dock_score_b)]])

        with torch.no_grad():
            score, logit = model(Batch.from_data_list([ga]), Batch.from_data_list([gb]), go_emb, dock)
            synergy_score = score.item()
            synergy_prob  = torch.sigmoid(logit).item()

        progress.progress(95)

        # Step 6: Show results
        with status:
            st.write("✅ Pipeline complete!")
        progress.progress(100)

        # 3D visualization
        with viz_placeholder.container():
            st.markdown("**Post-docking: Both drugs in protein binding pocket**")
            if pose_atoms_a or pose_atoms_b:
                show_docking_result(pdb_content, pose_atoms_a, pose_atoms_b, pdb_id)
                st.caption(f"🔵 {name_a or 'Drug A'} &nbsp; 🟠 {name_b or 'Drug B'} &nbsp; 🎨 Protein (spectrum) &nbsp; *Drag to rotate*")
            else:
                show_drugs_only_3d(smiles_a, smiles_b)
                st.caption("Docking poses unavailable — showing drug structures only")

        # Results
        st.markdown("---")
        st.markdown("### 📊 Results")

        if synergy_score > 4.0:   verdict, color = "✅ Strongly Synergistic", "green"
        elif synergy_score > 2.0: verdict, color = "⚠️ Mildly Synergistic", "orange"
        elif synergy_score > -1.0: verdict, color = "➖ Approximately Additive", "blue"
        else:                      verdict, color = "❌ Antagonistic", "red"

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Synergy Score", f"{synergy_score:.3f}", help="Loewe synergy score")
        m2.metric("Synergy Probability", f"{synergy_prob:.3f}")
        m3.metric(f"{name_a or 'Drug A'} Binding", f"{dock_score_a:.2f} kcal/mol")
        m4.metric(f"{name_b or 'Drug B'} Binding", f"{dock_score_b:.2f} kcal/mol")

        st.markdown(f"### Verdict: :{color}[{verdict}]")

        with st.expander("📋 Full docking report"):
            st.markdown(f"""
| Property | Value |
|----------|-------|
| Protein | {protein_name} |
| PDB ID | {pdb_id} |
| {name_a or 'Drug A'} docking score | {dock_score_a:.3f} kcal/mol |
| {name_b or 'Drug B'} docking score | {dock_score_b:.3f} kcal/mol |
| Synergy score (Loewe) | {synergy_score:.3f} |
| Synergy probability | {synergy_prob:.3f} |
| Verdict | {verdict} |
| Docking exhaustiveness | {exhaustiveness} |
| Binding box center | {[round(c,1) for c in center]} |
| Binding box size | {[round(s,1) for s in size]} Å |
            """)

        with st.expander("📖 How to interpret"):
            st.markdown("""
| Score | Meaning | Clinical Implication |
|-------|---------|---------------------|
| > 4.0 | Strongly Synergistic | Strong candidate for combination therapy |
| 2–4 | Mildly Synergistic | Modest benefit from combination |
| -1–2 | Approximately Additive | No significant interaction |
| < -1 | Antagonistic | Drugs may interfere — avoid combination |

The **Loewe synergy score** quantifies how much better (or worse) the drug 
combination performs versus independent drug action.
            """)

if __name__ == "__main__":
    pass
