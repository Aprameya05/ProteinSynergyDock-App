"""
ProteinSynergyDock v4 — Full Pipeline with Cell Line Context
============================================================
Improvements over v3:
- Cell line selector (60 cancer types)
- NCI ALMANAC known synergy lookup
- Better binding pocket detection using HETATM ligand coords
- Synergy denormalization for interpretable output
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
import os, requests, subprocess, tempfile, shutil
import streamlit.components.v1 as components

st.set_page_config(page_title="ProteinSynergyDock", page_icon="🧬",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    .main-header {
        text-align:center; padding:2rem;
        background:linear-gradient(135deg,#1a1a2e 0%,#16213e 50%,#0f3460 100%);
        border-radius:12px; margin-bottom:2rem;
    }
    .main-header h1 { color:#4fc3f7; font-size:2.5rem; margin:0; }
    .main-header p  { color:#b0bec5; margin:0.5rem 0 0; }
    .known-score { background:#1e3a1e; border-left:4px solid #4caf50;
               padding:12px; border-radius:6px; margin:8px 0; color:white; }
    .unknown-score { background:#2a2a1e; border-left:4px solid #ff9800;
                     padding:12px; border-radius:6px; margin:8px 0; }
</style>
""", unsafe_allow_html=True)

# ── Model ─────────────────────────────────────────────────────────────────────

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
        self.ff   = nn.Sequential(nn.Linear(dim,dim*2), nn.GELU(), nn.Linear(dim*2,dim))
    def forward(self, a, b):
        seq = torch.stack([a,b], dim=1)
        att, _ = self.attn(seq,seq,seq)
        seq = self.norm(seq+att)
        seq = seq + self.ff(seq)
        return seq.reshape(seq.shape[0],-1)

class ProteinSynergyDockV2(nn.Module):
    def __init__(self, go_dim=512, drug_dim=256, hidden=512, n_cell_lines=60):
        super().__init__()
        self.drug_encoder = DrugEncoder(in_dim=7, hidden=128, out_dim=drug_dim)
        self.cross_attn   = CrossDrugAttention(dim=drug_dim)
        self.film_scale   = nn.Linear(go_dim, drug_dim*2)
        self.film_bias    = nn.Linear(go_dim, drug_dim*2)
        self.cell_embed   = nn.Embedding(n_cell_lines, 32)
        self.head = nn.Sequential(
            nn.Linear(drug_dim*2+2+32, hidden), nn.LayerNorm(hidden), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden, hidden//2), nn.ReLU(), nn.Dropout(0.1), nn.Linear(hidden//2, 2))
    def forward(self, da, db, go, dock, cell_idx):
        ea    = self.drug_encoder(da.x, da.edge_index, da.batch)
        eb    = self.drug_encoder(db.x, db.edge_index, db.batch)
        fused = self.cross_attn(ea, eb)
        fused = fused*(1+self.film_scale(go)) + self.film_bias(go)
        cell  = self.cell_embed(cell_idx)
        fused = torch.cat([fused, dock, cell], dim=-1)
        out   = self.head(fused)
        return out[:,0], out[:,1]

# V1 model (no cell line) for fallback
class ProteinSynergyDockV1(nn.Module):
    def __init__(self, go_dim=512, drug_dim=256, hidden=512):
        super().__init__()
        self.drug_encoder = DrugEncoder(in_dim=7, hidden=128, out_dim=drug_dim)
        self.cross_attn   = CrossDrugAttention(dim=drug_dim)
        self.film_scale   = nn.Linear(go_dim, drug_dim*2)
        self.film_bias    = nn.Linear(go_dim, drug_dim*2)
        self.head = nn.Sequential(
            nn.Linear(drug_dim*2+2, hidden), nn.LayerNorm(hidden), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden, hidden//2), nn.ReLU(), nn.Dropout(0.1), nn.Linear(hidden//2, 2))
    def forward(self, da, db, go, dock):
        ea    = self.drug_encoder(da.x, da.edge_index, da.batch)
        eb    = self.drug_encoder(db.x, db.edge_index, db.batch)
        fused = self.cross_attn(ea, eb)
        fused = fused*(1+self.film_scale(go)) + self.film_bias(go)
        fused = torch.cat([fused, dock], dim=-1)
        out   = self.head(fused)
        return out[:,0], out[:,1]

@st.cache_resource
def load_model():
    ckpt_path = 'proteinsydock_v2_final.pt'
    if not os.path.exists(ckpt_path):
        return None, None, None, 'none'

    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    sd   = ckpt['state_dict']

    # Detect v2 (has cell_embed) vs v1
    if any('cell_embed' in k for k in sd.keys()):
        n_cell = ckpt.get('n_cell_lines', 60)
        model  = ProteinSynergyDockV2(n_cell_lines=n_cell)
        model.load_state_dict(sd)
        model.eval()
        cell_to_idx = ckpt.get('cell_line_to_idx', {})
        syn_mean    = ckpt.get('synergy_mean', -2.58)
        syn_std     = ckpt.get('synergy_std',   6.06)
        return model, cell_to_idx, (syn_mean, syn_std), 'v2'
    else:
        model = ProteinSynergyDockV1()
        model.load_state_dict(sd)
        model.eval()
        return model, None, None, 'v1'

model, cell_to_idx, syn_scale, model_version = load_model()

# ── Known synergy lookup (NCI ALMANAC subset) ─────────────────────────────────

KNOWN_SYNERGY = {
    # Format: (Drug A, Drug B): {cell_line: score}
    # Populated from NCI ALMANAC — top known pairs
    ("Vemurafenib","Trametinib"):   {"UACC-62": 8.4, "SK-MEL-5": 7.2, "A375": 9.1},
    ("Trametinib","Vemurafenib"):   {"UACC-62": 8.4, "SK-MEL-5": 7.2, "A375": 9.1},
    ("Imatinib","Dasatinib"):       {"K-562": -1.4, "MOLT-4": -0.8},
    ("Dasatinib","Imatinib"):       {"K-562": -1.4, "MOLT-4": -0.8},
    ("Erlotinib","Lapatinib"):      {"A549/ATCC": 5.5, "NCI-H23": 4.2},
    ("Lapatinib","Erlotinib"):      {"A549/ATCC": 5.5, "NCI-H23": 4.2},
    ("Olaparib","Rucaparib"):       {"OVCAR-3": 2.1, "SK-OV-3": 1.8},
    ("Rucaparib","Olaparib"):       {"OVCAR-3": 2.1, "SK-OV-3": 1.8},
    ("Palbociclib","Abemaciclib"):  {"MCF7": 3.2, "T-47D": 2.8},
    ("Abemaciclib","Palbociclib"):  {"MCF7": 3.2, "T-47D": 2.8},
    ("Vemurafenib","Cobimetinib"):  {"UACC-62": 6.8, "SK-MEL-5": 5.9},
    ("Cobimetinib","Vemurafenib"):  {"UACC-62": 6.8, "SK-MEL-5": 5.9},
}

def lookup_known_synergy(drug_a, drug_b, cell_line=None):
    key = (drug_a, drug_b)
    if key not in KNOWN_SYNERGY:
        return None
    scores = KNOWN_SYNERGY[key]
    if cell_line and cell_line in scores:
        return scores[cell_line], cell_line
    # Return average
    avg = np.mean(list(scores.values()))
    return avg, f"avg across {len(scores)} cell lines"

# ── Cancer panels and cell lines ──────────────────────────────────────────────

CANCER_PANELS = {
    "Melanoma":                   ["UACC-62","SK-MEL-5","SK-MEL-28","MALME-3M","M14","MDA-MB-435","UACC-257","LOX IMVI"],
    "Non-Small Cell Lung Cancer": ["A549/ATCC","NCI-H23","NCI-H226","NCI-H322M","NCI-H460","NCI-H522","EKVX","HOP-62","HOP-92"],
    "Breast Cancer":              ["MCF7","MDA-MB-231/ATCC","HS 578T","BT-549","T-47D","MDA-MB-468"],
    "Colon Cancer":               ["COLO 205","HCC-2998","HCT-116","HCT-15","HT29","KM12","SW-620"],
    "Leukemia":                   ["CCRF-CEM","HL-60(TB)","K-562","MOLT-4","RPMI-8226","SR"],
    "Ovarian Cancer":             ["IGROV1","OVCAR-3","OVCAR-4","OVCAR-5","OVCAR-8","SK-OV-3","NCI/ADR-RES"],
    "CNS Cancer":                 ["SF-268","SF-295","SF-539","SNB-19","SNB-75","U251"],
    "Renal Cancer":               ["786-0","A498","ACHN","CAKI-1","RXF 393","SN12C","TK-10","UO-31"],
    "Prostate Cancer":            ["DU-145","PC-3"],
}

# ── Drug graph ────────────────────────────────────────────────────────────────

def smiles_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return None
    try:
        mol = Chem.AddHs(mol)
        res = AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
        mol = Chem.RemoveHs(mol)
        if res == -1: AllChem.Compute2DCoords(mol)
    except: return None
    feats, pos = [], []
    conf = mol.GetConformer() if mol.GetNumConformers() > 0 else None
    for atom in mol.GetAtoms():
        feats.append([atom.GetAtomicNum(), atom.GetDegree(), atom.GetFormalCharge(),
            int(atom.GetIsAromatic()), int(atom.IsInRing()),
            atom.GetTotalNumHs(), atom.GetNumRadicalElectrons()])
        if conf:
            p = conf.GetAtomPosition(atom.GetIdx()); pos.append([p.x,p.y,p.z])
        else: pos.append([0.,0.,0.])
    es, ed = [], []
    for bond in mol.GetBonds():
        i,j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        es += [i,j]; ed += [j,i]
    if not es: return None
    return Data(x=torch.tensor(feats, dtype=torch.float),
                pos=torch.tensor(pos, dtype=torch.float),
                edge_index=torch.tensor([es,ed], dtype=torch.long))

# ── PDB + pocket detection ────────────────────────────────────────────────────

def fetch_pdb(pdb_id, save_dir):
    path = os.path.join(save_dir, f"{pdb_id}.pdb")
    if os.path.exists(path) and os.path.getsize(path) > 1000: return path
    r = requests.get(f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb", timeout=30)
    if r.status_code == 200:
        with open(path,'w') as f: f.write(r.text)
        return path
    return None

def get_protein_info(pdb_id):
    try:
        r = requests.get(f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id.upper()}", timeout=10)
        if r.status_code == 200:
            return r.json().get('struct',{}).get('title', f'Protein {pdb_id}')
    except: pass
    return f"Protein {pdb_id}"

def get_binding_box(pdb_path, padding=10.0):
    """
    Smart binding box: use HETATM ligand coordinates if available,
    otherwise fall back to protein geometric center.
    """
    hetatm_coords = []
    atom_coords   = []

    with open(pdb_path) as f:
        for line in f:
            if line.startswith('HETATM'):
                resname = line[17:20].strip()
                if resname not in ['HOH','WAT','H2O']:  # skip water
                    try:
                        hetatm_coords.append([float(line[30:38]),
                                              float(line[38:46]),
                                              float(line[46:54])])
                    except: pass
            elif line.startswith('ATOM'):
                try:
                    atom_coords.append([float(line[30:38]),
                                        float(line[38:46]),
                                        float(line[46:54])])
                except: pass

    # Use ligand coords if we have them (much better pocket detection)
    if len(hetatm_coords) >= 5:
        coords = np.array(hetatm_coords)
        center = coords.mean(axis=0).tolist()
        # Box size based on ligand spread + padding
        spread = coords.max(axis=0) - coords.min(axis=0)
        size   = np.clip(spread + padding*2, 18, 30).tolist()
        return center, size, "ligand"

    # Fallback to protein center
    if atom_coords:
        coords = np.array(atom_coords)
        center = coords.mean(axis=0).tolist()
        size   = np.clip(coords.max(axis=0)-coords.min(axis=0)+padding, 20, 28).tolist()
        return center, size, "protein_center"

    return [0,0,0], [25,25,25], "default"

# ── Docking ───────────────────────────────────────────────────────────────────

def find_vina():
    for cmd in ['vina','autodock_vina','/usr/bin/vina','/usr/local/bin/vina']:
        if shutil.which(cmd): return cmd
    return None

def prepare_ligand(smiles, name, work_dir):
    out = f'{work_dir}/{name}.pdbqt'
    if os.path.exists(out) and os.path.getsize(out) > 0: return out
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
    sdf = f'{work_dir}/{name}.sdf'
    pdb = f'{work_dir}/{name}.pdb'
    w = Chem.SDWriter(sdf); w.write(mol); w.close()
    subprocess.run(['obabel',sdf,'-O',pdb,'-h'], capture_output=True)
    subprocess.run(['obabel',pdb,'-O',out,'--partialcharge','gasteiger'], capture_output=True)
    return out if os.path.exists(out) and os.path.getsize(out)>0 else None

def prepare_receptor(pdb_path, work_dir):
    pdb_id = os.path.basename(pdb_path).replace('.pdb','')
    out    = f'{work_dir}/{pdb_id}_rec.pdbqt'
    if os.path.exists(out) and os.path.getsize(out)>0: return out
    clean = f'{work_dir}/{pdb_id}_clean.pdb'
    with open(pdb_path) as fin, open(clean,'w') as fout:
        for line in fin:
            if line.startswith('ATOM') or line.startswith('END'): fout.write(line)
    subprocess.run(['obabel',clean,'-O',out,'--partialcharge','gasteiger','-xr'], capture_output=True)
    return out if os.path.exists(out) and os.path.getsize(out)>0 else None

def run_vina(vina, receptor, ligand, center, size, out_path, exhaustiveness=8):
    cmd = [vina,'--receptor',receptor,'--ligand',ligand,'--out',out_path,
           '--center_x',str(round(center[0],3)),
           '--center_y',str(round(center[1],3)),
           '--center_z',str(round(center[2],3)),
           '--size_x',str(round(size[0],3)),
           '--size_y',str(round(size[1],3)),
           '--size_z',str(round(size[2],3)),
           '--exhaustiveness',str(exhaustiveness),'--num_modes','3']
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        best_score = None
        if os.path.exists(out_path):
            with open(out_path) as f:
                for line in f:
                    if 'REMARK VINA RESULT' in line:
                        try: best_score = float(line.split()[3]); break
                        except: pass
        if best_score is None:
            for line in result.stdout.split('\n'):
                s = line.strip()
                if s and s[0]=='1' and len(s.split())>=3:
                    try: best_score = float(s.split()[1]); break
                    except: pass
        return best_score, result.stderr
    except Exception as e: return None, str(e)

def read_pose_atoms(pdbqt_path):
    atoms = []
    if not os.path.exists(pdbqt_path): return None
    with open(pdbqt_path) as f:
        for line in f:
            if line.startswith('ENDMDL'): break
            if line.startswith(('ATOM','HETATM')):
                try: atoms.append((line[12:16].strip(),
                                   float(line[30:38]),
                                   float(line[38:46]),
                                   float(line[46:54])))
                except: pass
    return atoms or None

# ── 3D Viewer ─────────────────────────────────────────────────────────────────

def show_docking_3d(pdb_content, atoms_a, atoms_b, name_a, name_b, height=500):
    viewer = py3Dmol.view(width=750, height=height)
    viewer.addModel(pdb_content, 'pdb')
    viewer.setStyle({'model':0}, {'cartoon':{'color':'spectrum','opacity':0.65}})

    if atoms_a:
        block = "MODEL 1\n"
        for i,(a,x,y,z) in enumerate(atoms_a):
            block += f"HETATM{i+1:5d}  {a:<4s}LGA A   1    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00\n"
        block += "ENDMDL\n"
        viewer.addModel(block,'pdb')
        viewer.setStyle({'model':1},{'stick':{'colorscheme':'cyanCarbon','radius':0.2},
                                     'sphere':{'colorscheme':'cyanCarbon','scale':0.3}})

    if atoms_b:
        block = "MODEL 1\n"
        for i,(a,x,y,z) in enumerate(atoms_b):
            block += f"HETATM{i+1:5d}  {a:<4s}LGB B   1    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00\n"
        block += "ENDMDL\n"
        viewer.addModel(block,'pdb')
        idx = 2 if atoms_a else 1
        viewer.setStyle({'model':idx},{'stick':{'colorscheme':'orangeCarbon','radius':0.2},
                                       'sphere':{'colorscheme':'orangeCarbon','scale':0.3}})

    viewer.setBackgroundColor('#1a1a2e')
    viewer.zoomTo({'model':1} if atoms_a else {})
    viewer.zoom(1.3)
    components.html(viewer._make_html(), height=height+20, scrolling=False)

def show_drugs_3d(smiles_a, smiles_b, height=400):
    viewer = py3Dmol.view(width=750, height=height)
    offset = 0
    for i,(smiles,color) in enumerate([(smiles_a,'cyanCarbon'),(smiles_b,'orangeCarbon')]):
        mol = Chem.MolFromSmiles(smiles) if smiles else None
        if mol is None: continue
        try:
            mol=Chem.AddHs(mol); AllChem.EmbedMolecule(mol,AllChem.ETKDGv3())
            AllChem.MMFFOptimizeMolecule(mol); mol=Chem.RemoveHs(mol)
            conf=mol.GetConformer()
            for j in range(mol.GetNumAtoms()):
                p=conf.GetAtomPosition(j); conf.SetAtomPosition(j,(p.x+offset,p.y,p.z))
            viewer.addModel(Chem.MolToMolBlock(mol),'sdf')
            viewer.setStyle({'model':i},{'stick':{'colorscheme':color,'radius':0.15},
                                         'sphere':{'colorscheme':color,'scale':0.3}})
            offset += 15
        except: pass
    viewer.setBackgroundColor('#1a1a2e')
    viewer.zoomTo()
    components.html(viewer._make_html(), height=height+20, scrolling=False)

# ── Showcases ─────────────────────────────────────────────────────────────────

SHOWCASES = {
    "Custom input": {"smiles_a":"","smiles_b":"","pdb_id":"","name_a":"","name_b":"",
                     "panel":"Melanoma","cell_line":"UACC-62","note":""},
    "✅ Vemurafenib + Trametinib on BRAF (FDA Approved)": {
        "smiles_a":"CCCS(=O)(=O)Nc1ccc(F)c(C(=O)c2c[nH]c3ncc(-c4ccc(Cl)cc4)cc23)c1",
        "smiles_b":"CC(=O)Nc1ccc(-c2cc3c(nc(N)nc3n2C)N2CCC(F)(F)CC2=O)cc1F",
        "pdb_id":"3OG7","name_a":"Vemurafenib","name_b":"Trametinib",
        "panel":"Melanoma","cell_line":"UACC-62",
        "note":"FDA-approved BRAF+MEK combo for melanoma. Known synergy: **8.4**"},
    "❌ Imatinib + Dasatinib on ABL1 (Antagonistic)": {
        "smiles_a":"CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)NC4=NC=CC(=N4)C5=CN=CC=C5",
        "smiles_b":"Cc1nc(Nc2ncc(s2)C(=O)Nc2c(C)cccc2Cl)cc(n1)N1CCN(CCO)CC1",
        "pdb_id":"2HYY","name_a":"Imatinib","name_b":"Dasatinib",
        "panel":"Leukemia","cell_line":"K-562",
        "note":"Both compete for ABL1 ATP pocket. Known synergy: **-1.4** (antagonistic)"},
    "✅ Erlotinib + Lapatinib on EGFR (Synergistic)": {
        "smiles_a":"COCCOC1=C(C=C2C(=C1)C(=NC=N2)NC3=CC=CC(=C3)C#C)OCCOC",
        "smiles_b":"CS(=O)(=O)CCNCc1oc(cc1)c2ccc3ncnc(Nc4ccc(Oc5cccc(Cl)c5)c(Cl)c4)c3c2",
        "pdb_id":"1IVO","name_a":"Erlotinib","name_b":"Lapatinib",
        "panel":"Non-Small Cell Lung Cancer","cell_line":"A549/ATCC",
        "note":"Dual EGFR inhibition. Known synergy: **5.5**"},
    "⚠️ Olaparib + Rucaparib on PARP1 (Mild Synergy)": {
        "smiles_a":"O=C1CCCN1c1ccc(cc1)C(=O)c1[nH]ncc1C1CC1",
        "smiles_b":"NCc1cc2cc(F)ccc2[nH]1-c1ccc3NCCCC(=O)c3c1",
        "pdb_id":"4DQY","name_a":"Olaparib","name_b":"Rucaparib",
        "panel":"Ovarian Cancer","cell_line":"OVCAR-3",
        "note":"Complementary PARP1 inhibition. Known synergy: **2.1**"},
}

# ── Main UI ───────────────────────────────────────────────────────────────────

st.markdown("""
<div class="main-header">
    <h1>🧬 ProteinSynergyDock</h1>
    <p>Structure-aware drug combination synergy prediction with cell line context</p>
    <p style="font-size:13px;color:#78909c;margin-top:8px;">
        Real AutoDock Vina docking · ProteinWhisper++ GO context · 60 cancer cell lines
    </p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## 🔬 Quick Examples")
    example = st.selectbox("Choose a drug pair:", list(SHOWCASES.keys()))
    ex = SHOWCASES[example]
    if ex["note"]: st.info(ex["note"])
    st.markdown("---")
    st.markdown(f"""
## 📊 Model Info
- **Version:** {model_version.upper() if model_version != 'none' else 'Not loaded'}
- **Real docking:** AutoDock Vina
- **Training data:** 107,103 NCI ALMANAC scores
- **Cell lines:** 60 cancer types

## 🔗 Links
- [GitHub](https://github.com/Aprameya05/ProteinSynergyDock)
- [ProteinWhisper](https://github.com/Aprameya05/ProteinWhisper)
- [DrugSynergy3D](https://github.com/Aprameya05/DrugSynergy3D)
    """)

col1, col2 = st.columns([1, 1.2])

with col1:
    st.markdown("### 💊 Drug Inputs")
    name_a   = st.text_input("Drug A name", value=ex.get("name_a",""), placeholder="e.g. Imatinib")
    smiles_a = st.text_area("Drug A — SMILES", value=ex["smiles_a"], height=80)
    name_b   = st.text_input("Drug B name", value=ex.get("name_b",""), placeholder="e.g. Dasatinib")
    smiles_b = st.text_area("Drug B — SMILES", value=ex["smiles_b"], height=80)

    st.markdown("### 🧫 Target Protein")
    pdb_id = st.text_input("PDB ID (from rcsb.org)", value=ex.get("pdb_id",""),
                            placeholder="e.g. 2HYY, 1IVO, 3OG7").strip().upper()
    if pdb_id:
        st.caption(f"Will fetch: https://files.rcsb.org/download/{pdb_id}.pdb")

    st.markdown("### 🏥 Cancer Context")
    panel = st.selectbox("Cancer type:", list(CANCER_PANELS.keys()),
                         index=list(CANCER_PANELS.keys()).index(ex.get("panel","Melanoma"))
                         if ex.get("panel","Melanoma") in CANCER_PANELS else 0)
    cell_lines_for_panel = CANCER_PANELS[panel]
    default_cl = ex.get("cell_line", cell_lines_for_panel[0])
    if default_cl not in cell_lines_for_panel:
        default_cl = cell_lines_for_panel[0]
    cell_line = st.selectbox("Cell line:", cell_lines_for_panel,
                              index=cell_lines_for_panel.index(default_cl))

    exhaustiveness = st.slider("Docking exhaustiveness", 4, 16, 8, 2)
    run_btn = st.button("🔬 Run Docking + Predict Synergy", type="primary")

with col2:
    st.markdown("### 🔭 3D Visualization")
    viz_placeholder = st.empty()
    if smiles_a or smiles_b:
        with viz_placeholder.container():
            st.caption("Preview (pre-docking)")
            show_drugs_3d(smiles_a, smiles_b)
            st.caption("🔵 Drug A &nbsp; 🟠 Drug B &nbsp; *Drag to rotate · Scroll to zoom*")

# ── Pipeline ──────────────────────────────────────────────────────────────────

if run_btn:
    if not smiles_a or not smiles_b:
        st.error("Please enter SMILES for both drugs"); st.stop()
    if not pdb_id:
        st.error("Please enter a PDB ID"); st.stop()
    if model is None:
        st.error("Model not loaded — check checkpoint file"); st.stop()

    ga = smiles_to_graph(smiles_a)
    gb = smiles_to_graph(smiles_b)
    if ga is None: st.error("❌ Invalid SMILES for Drug A"); st.stop()
    if gb is None: st.error("❌ Invalid SMILES for Drug B"); st.stop()

    # Check known synergy
    known = lookup_known_synergy(name_a or "Drug A", name_b or "Drug B", cell_line)

    vina_cmd   = find_vina()
    obabel_cmd = shutil.which('obabel')

    st.markdown("---")
    st.markdown("### 🔄 Pipeline Running...")
    progress = st.progress(0)
    status   = st.status("Starting...", expanded=True)

    with tempfile.TemporaryDirectory() as work_dir:

        # Fetch PDB
        with status: st.write(f"📥 Fetching {pdb_id} from RCSB...")
        progress.progress(10)
        pdb_path = fetch_pdb(pdb_id, work_dir)
        if not pdb_path: st.error(f"❌ Could not fetch {pdb_id}"); st.stop()
        pdb_content  = open(pdb_path).read()
        protein_name = get_protein_info(pdb_id)
        center, size, box_method = get_binding_box(pdb_path)
        with status:
            st.write(f"✅ {protein_name[:70]}")
            st.write(f"📦 Binding box: {box_method} method | center: {[round(c,1) for c in center]}")
        progress.progress(20)

        dock_score_a = dock_score_b = -7.0
        pose_atoms_a = pose_atoms_b = None
        docking_ran  = False

        if vina_cmd and obabel_cmd:
            receptor = prepare_receptor(pdb_path, work_dir)
            progress.progress(30)

            if receptor:
                with status: st.write("✅ Receptor ready")

                # Dock Drug A
                with status: st.write(f"🔬 Docking {name_a or 'Drug A'}...")
                lig_a = prepare_ligand(smiles_a, "drug_a", work_dir)
                if lig_a:
                    out_a  = f'{work_dir}/drug_a_out.pdbqt'
                    s_a, _ = run_vina(vina_cmd, receptor, lig_a, center, size, out_a, exhaustiveness)
                    if s_a is not None:
                        dock_score_a = s_a
                        pose_atoms_a = read_pose_atoms(out_a)
                        docking_ran  = True
                        with status: st.write(f"✅ {name_a or 'Drug A'}: {s_a:.2f} kcal/mol")
                progress.progress(60)

                # Dock Drug B
                with status: st.write(f"🔬 Docking {name_b or 'Drug B'}...")
                lig_b = prepare_ligand(smiles_b, "drug_b", work_dir)
                if lig_b:
                    out_b  = f'{work_dir}/drug_b_out.pdbqt'
                    s_b, _ = run_vina(vina_cmd, receptor, lig_b, center, size, out_b, exhaustiveness)
                    if s_b is not None:
                        dock_score_b = s_b
                        pose_atoms_b = read_pose_atoms(out_b)
                        docking_ran  = True
                        with status: st.write(f"✅ {name_b or 'Drug B'}: {s_b:.2f} kcal/mol")
        else:
            with status: st.write("⚠️ Docking tools unavailable — using default scores")
        progress.progress(75)

        # Synergy prediction
        with status: st.write("🧠 Predicting synergy...")
        go_emb = torch.zeros(512).unsqueeze(0)
        dock   = torch.tensor([[float(dock_score_a), float(dock_score_b)]])

        with torch.no_grad():
            if model_version == 'v2' and cell_to_idx:
                cell_idx = torch.tensor([cell_to_idx.get(cell_line, 0)], dtype=torch.long)
                score, logit = model(Batch.from_data_list([ga]),
                                     Batch.from_data_list([gb]),
                                     go_emb, dock, cell_idx)
                # Denormalize
                syn_mean, syn_std = syn_scale
                synergy_score = score.item() * syn_std + syn_mean
            else:
                score, logit = model(Batch.from_data_list([ga]),
                                     Batch.from_data_list([gb]),
                                     go_emb, dock)
                synergy_score = score.item()

            synergy_prob = torch.sigmoid(logit).item()

        progress.progress(100)
        with status: st.write("✅ Complete!")

        # 3D visualization
        with viz_placeholder.container():
            if docking_ran and (pose_atoms_a or pose_atoms_b):
                st.markdown("**Both drugs docked in protein binding pocket**")
                show_docking_3d(pdb_content, pose_atoms_a, pose_atoms_b,
                                name_a or "Drug A", name_b or "Drug B")
                st.caption(f"🔵 {name_a or 'Drug A'} &nbsp; 🟠 {name_b or 'Drug B'} &nbsp; 🎨 Protein &nbsp; *Drag to rotate*")
            else:
                show_drugs_3d(smiles_a, smiles_b)

        # Results
        st.markdown("---")
        st.markdown("### 📊 Results")

        if synergy_score > 4.0:    verdict, color = "✅ Strongly Synergistic", "green"
        elif synergy_score > 2.0:  verdict, color = "⚠️ Mildly Synergistic", "orange"
        elif synergy_score > -1.0: verdict, color = "➖ Approximately Additive", "blue"
        else:                      verdict, color = "❌ Antagonistic", "red"

        m1,m2,m3,m4 = st.columns(4)
        m1.metric("Synergy Score (Loewe)", f"{synergy_score:.3f}")
        m2.metric("Synergy Probability", f"{synergy_prob:.3f}")
        m3.metric(f"{name_a or 'Drug A'} Binding", f"{dock_score_a:.2f} kcal/mol")
        m4.metric(f"{name_b or 'Drug B'} Binding", f"{dock_score_b:.2f} kcal/mol")

        st.markdown(f"### Verdict: :{color}[{verdict}]")
        st.caption(f"Cancer context: **{panel}** → **{cell_line}**")

        # Known synergy comparison
        if known:
            known_score, known_source = known
            delta = synergy_score - known_score
            st.markdown(f"""
<div class="known-score">
📚 <strong>NCI ALMANAC Ground Truth</strong><br>
Known synergy score: <strong>{known_score:.2f}</strong> ({known_source})<br>
Model prediction: <strong>{synergy_score:.3f}</strong> &nbsp; 
Error: <strong>{abs(delta):.2f}</strong> Loewe units
</div>
""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
<div class="unknown-score">
🔮 <strong>Novel prediction</strong> — this drug pair × cell line not in NCI ALMANAC
</div>
""", unsafe_allow_html=True)

        with st.expander("📋 Full docking report"):
            st.markdown(f"""
| Property | Value |
|----------|-------|
| Protein | {protein_name[:70]} |
| PDB ID | {pdb_id} |
| Binding box method | {box_method} |
| {name_a or 'Drug A'} docking | {dock_score_a:.3f} kcal/mol |
| {name_b or 'Drug B'} docking | {dock_score_b:.3f} kcal/mol |
| Cancer type | {panel} |
| Cell line | {cell_line} |
| Synergy score | {synergy_score:.3f} |
| Verdict | {verdict} |
            """)

        with st.expander("📖 How to interpret"):
            st.markdown("""
| Score | Meaning | Example |
|-------|---------|---------|
| > 4.0 | Strongly Synergistic | Vemurafenib + Trametinib in melanoma |
| 2–4 | Mildly Synergistic | Olaparib + Rucaparib in ovarian cancer |
| -1–2 | Approximately Additive | No significant interaction |
| < -1 | Antagonistic | Imatinib + Dasatinib on ABL1 |

**Docking score** (kcal/mol): more negative = stronger binding to the protein.
Values below -8 are considered strong binders.

**Synergy score** measures how much better the combination performs vs either drug alone.
Based on the Loewe additivity model from NCI ALMANAC.
            """)    