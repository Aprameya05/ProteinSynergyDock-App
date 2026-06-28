"""
ProteinSynergyDock — Full Auto-Docking Pipeline
Uses AutoDock Vina binary via subprocess
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
import shutil
import streamlit.components.v1 as components

st.set_page_config(page_title="ProteinSynergyDock", page_icon="🧬",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    .main-header {
        text-align: center; padding: 2rem;
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        border-radius: 12px; margin-bottom: 2rem;
    }
    .main-header h1 { color: #4fc3f7; font-size: 2.5rem; margin: 0; }
    .main-header p  { color: #b0bec5; margin: 0.5rem 0 0; }
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
    if os.path.exists('proteinsydock_v2_final.pt'):
        ckpt = torch.load('proteinsydock_v2_final.pt', map_location='cpu', weights_only=False)
        model.load_state_dict(ckpt['state_dict'])
        model.eval()
        return model, ckpt['pearson_r'], ckpt['auroc']
    return model, 0.0, 0.0

model, best_r, best_auroc = load_model()

# ── Find vina binary ──────────────────────────────────────────────────────────

def find_vina():
    for cmd in ['vina', 'autodock_vina', '/usr/bin/vina', '/usr/local/bin/vina']:
        if shutil.which(cmd):
            return cmd
    return None

def find_obabel():
    return shutil.which('obabel')

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

# ── PDB ───────────────────────────────────────────────────────────────────────

def fetch_pdb(pdb_id, save_dir):
    path = os.path.join(save_dir, f"{pdb_id}.pdb")
    if os.path.exists(path) and os.path.getsize(path) > 1000:
        return path
    r = requests.get(f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb", timeout=30)
    if r.status_code == 200:
        with open(path, 'w') as f: f.write(r.text)
        return path
    return None

def get_protein_info(pdb_id):
    try:
        r = requests.get(f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id.upper()}", timeout=10)
        if r.status_code == 200:
            return r.json().get('struct', {}).get('title', f'Protein {pdb_id}')
    except: pass
    return f"Protein {pdb_id}"

def get_binding_box(pdb_path, padding=10.0):
    coords = []
    with open(pdb_path) as f:
        for line in f:
            if line.startswith(('ATOM', 'HETATM')):
                try: coords.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
                except: pass
    if not coords: return [0,0,0], [30,30,30]
    coords = np.array(coords)
    center = coords.mean(axis=0).tolist()
    size   = np.clip(coords.max(axis=0) - coords.min(axis=0) + padding, 20, 30).tolist()
    return center, size

# ── Docking ───────────────────────────────────────────────────────────────────

def prepare_ligand(smiles, name, work_dir, obabel):
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
    sdf   = os.path.join(work_dir, f"{name}.sdf")
    pdb   = os.path.join(work_dir, f"{name}.pdb")
    pdbqt = os.path.join(work_dir, f"{name}.pdbqt")
    w = Chem.SDWriter(sdf); w.write(mol); w.close()
    subprocess.run([obabel, sdf, '-O', pdb, '-h'], capture_output=True)
    subprocess.run([obabel, pdb, '-O', pdbqt, '--partialcharge', 'gasteiger'], capture_output=True)
    return pdbqt if os.path.exists(pdbqt) and os.path.getsize(pdbqt) > 0 else None

def prepare_receptor(pdb_path, work_dir, obabel):
    pdb_id = os.path.basename(pdb_path).replace('.pdb', '')
    pdbqt  = os.path.join(work_dir, f"{pdb_id}_rec.pdbqt")
    if os.path.exists(pdbqt) and os.path.getsize(pdbqt) > 0: return pdbqt

    # Clean PDB first - keep only ATOM records (remove HETATM, waters)
    clean_pdb = os.path.join(work_dir, f"{pdb_id}_clean.pdb")
    with open(pdb_path) as fin, open(clean_pdb, 'w') as fout:
        for line in fin:
            if line.startswith('ATOM') or line.startswith('END'):
                fout.write(line)

    # Convert cleaned PDB to PDBQT
    subprocess.run([obabel, clean_pdb, '-O', pdbqt,
                    '--partialcharge', 'gasteiger', '-xr'],
                   capture_output=True)
    return pdbqt if os.path.exists(pdbqt) and os.path.getsize(pdbqt) > 0 else None

def run_vina(vina, receptor, ligand, center, size, out_path, exhaustiveness=8):
    cmd = [vina,
        '--receptor', receptor, '--ligand', ligand, '--out', out_path,
        '--center_x', str(round(center[0],3)),
        '--center_y', str(round(center[1],3)),
        '--center_z', str(round(center[2],3)),
        '--size_x',   str(round(size[0],3)),
        '--size_y',   str(round(size[1],3)),
        '--size_z',   str(round(size[2],3)),
        '--exhaustiveness', str(exhaustiveness),
        '--num_modes', '3',
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        stdout = result.stdout
        stderr = result.stderr

        # Parse score from stdout table
        best_score = None
        for line in stdout.split('\n'):
            stripped = line.strip()
            if stripped and stripped[0] == '1' and len(stripped.split()) >= 3:
                try:
                    best_score = float(stripped.split()[1])
                    break
                except: pass

        # Parse score from output pdbqt
        if best_score is None and os.path.exists(out_path):
            with open(out_path) as f:
                for line in f:
                    if 'REMARK VINA RESULT' in line:
                        try: best_score = float(line.split()[3]); break
                        except: pass

        return best_score, stdout, stderr
    except Exception as e:
        return None, "", str(e)

def read_pose_atoms(pdbqt_path):
    atoms = []
    if not os.path.exists(pdbqt_path): return None
    with open(pdbqt_path) as f:
        for line in f:
            if line.startswith('ENDMDL'): break
            if line.startswith(('ATOM','HETATM')):
                try: atoms.append((line[12:16].strip(), float(line[30:38]),
                                   float(line[38:46]), float(line[46:54])))
                except: pass
    return atoms or None

# ── 3D Viewer ─────────────────────────────────────────────────────────────────

def show_docking_3d(pdb_content, atoms_a, atoms_b, name_a, name_b, height=500):
    viewer = py3Dmol.view(width=750, height=height)
    viewer.addModel(pdb_content, 'pdb')
    viewer.setStyle({'model': 0}, {'cartoon': {'color': 'spectrum', 'opacity': 0.85}})

    if atoms_a:
        block = "MODEL 1\n"
        for i,(a,x,y,z) in enumerate(atoms_a):
            block += f"HETATM{i+1:5d}  {a:<4s}LGA A   1    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00\n"
        block += "ENDMDL\n"
        viewer.addModel(block, 'pdb')
        viewer.setStyle({'model': 1}, {'stick': {'colorscheme':'cyanCarbon','radius':0.2},
                                       'sphere': {'colorscheme':'cyanCarbon','scale':0.3}})

    if atoms_b:
        block = "MODEL 1\n"
        for i,(a,x,y,z) in enumerate(atoms_b):
            block += f"HETATM{i+1:5d}  {a:<4s}LGB B   1    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00\n"
        block += "ENDMDL\n"
        viewer.addModel(block, 'pdb')
        idx = 2 if atoms_a else 1
        viewer.setStyle({'model': idx}, {'stick': {'colorscheme':'orangeCarbon','radius':0.2},
                                          'sphere': {'colorscheme':'orangeCarbon','scale':0.3}})

    viewer.setBackgroundColor('#1a1a2e')
    viewer.zoomTo({'model': 1} if atoms_a else {})
    viewer.zoom(1.3)
    components.html(viewer._make_html(), height=height+20, scrolling=False)

def show_drugs_3d(smiles_a, smiles_b, height=400):
    viewer = py3Dmol.view(width=750, height=height)
    offset = 0
    for i,(smiles,color) in enumerate([(smiles_a,'cyanCarbon'),(smiles_b,'orangeCarbon')]):
        mol = Chem.MolFromSmiles(smiles) if smiles else None
        if mol is None: continue
        try:
            mol = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
            AllChem.MMFFOptimizeMolecule(mol)
            mol = Chem.RemoveHs(mol)
            conf = mol.GetConformer()
            for j in range(mol.GetNumAtoms()):
                p = conf.GetAtomPosition(j)
                conf.SetAtomPosition(j, (p.x+offset, p.y, p.z))
            viewer.addModel(Chem.MolToMolBlock(mol), 'sdf')
            viewer.setStyle({'model': i}, {'stick':{'colorscheme':color,'radius':0.15},
                                           'sphere':{'colorscheme':color,'scale':0.3}})
            offset += 15
        except: pass
    viewer.setBackgroundColor('#1a1a2e')
    viewer.zoomTo()
    components.html(viewer._make_html(), height=height+20, scrolling=False)

# ── Showcases ─────────────────────────────────────────────────────────────────

SHOWCASES = {
    "Custom input": {"smiles_a":"","smiles_b":"","pdb_id":"","name_a":"","name_b":"","note":""},
    "✅ Vemurafenib + Trametinib on BRAF (Approved Combo)": {
        "smiles_a":"CCCS(=O)(=O)Nc1ccc(F)c(C(=O)c2c[nH]c3ncc(-c4ccc(Cl)cc4)cc23)c1",
        "smiles_b":"CC(=O)Nc1ccc(-c2cc3c(nc(N)nc3n2C)N2CCC(F)(F)CC2=O)cc1F",
        "pdb_id":"3OG7","name_a":"Vemurafenib","name_b":"Trametinib",
        "note":"FDA-approved BRAF+MEK combination for melanoma. Known synergy: **8.4**"},
    "❌ Imatinib + Dasatinib on ABL1 (Antagonistic)": {
        "smiles_a":"CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)NC4=NC=CC(=N4)C5=CN=CC=C5",
        "smiles_b":"Cc1nc(Nc2ncc(s2)C(=O)Nc2c(C)cccc2Cl)cc(n1)N1CCN(CCO)CC1",
        "pdb_id":"2HYY","name_a":"Imatinib","name_b":"Dasatinib",
        "note":"Both compete for ABL1 ATP pocket. Known synergy: **-1.4** (antagonistic)"},
    "✅ Erlotinib + Lapatinib on EGFR (Synergistic)": {
        "smiles_a":"COCCOC1=C(C=C2C(=C1)C(=NC=N2)NC3=CC=CC(=C3)C#C)OCCOC",
        "smiles_b":"CS(=O)(=O)CCNCc1oc(cc1)c2ccc3ncnc(Nc4ccc(Oc5cccc(Cl)c5)c(Cl)c4)c3c2",
        "pdb_id":"1IVO","name_a":"Erlotinib","name_b":"Lapatinib",
        "note":"Dual EGFR inhibition. Known synergy: **5.5**"},
    "⚠️ Olaparib + Rucaparib on PARP1 (Mild Synergy)": {
        "smiles_a":"O=C1CCCN1c1ccc(cc1)C(=O)c1[nH]ncc1C1CC1",
        "smiles_b":"NCc1cc2cc(F)ccc2[nH]1-c1ccc3NCCCC(=O)c3c1",
        "pdb_id":"4DQY","name_a":"Olaparib","name_b":"Rucaparib",
        "note":"Complementary PARP1 inhibition. Known synergy: **2.1**"},
}

# ── UI ────────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="main-header">
    <h1>🧬 ProteinSynergyDock</h1>
    <p>Automatic molecular docking + drug combination synergy prediction</p>
    <p style="font-size:13px;color:#78909c;margin-top:8px;">
        Input two drugs + a protein → auto-docking → 3D visualization → synergy score
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
## 📊 Model Stats
- **Pearson r:** {best_r:.4f}
- **AUROC:** {best_auroc:.4f}
- **Docking:** AutoDock Vina
- **Data:** 231 NCI ALMANAC scores

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

    ga = smiles_to_graph(smiles_a)
    gb = smiles_to_graph(smiles_b)
    if ga is None: st.error("❌ Invalid SMILES for Drug A"); st.stop()
    if gb is None: st.error("❌ Invalid SMILES for Drug B"); st.stop()

    vina_cmd   = find_vina()
    obabel_cmd = find_obabel()

    st.markdown("---")
    st.markdown("### 🔄 Pipeline Running...")
    progress = st.progress(0)
    status   = st.status("Starting...", expanded=True)

    with tempfile.TemporaryDirectory() as work_dir:

        # Step 1: Fetch PDB
        with status: st.write(f"📥 Fetching {pdb_id} from RCSB...")
        progress.progress(10)
        pdb_path = fetch_pdb(pdb_id, work_dir)
        if not pdb_path: st.error(f"❌ Could not fetch PDB {pdb_id}"); st.stop()
        pdb_content  = open(pdb_path).read()
        protein_name = get_protein_info(pdb_id)
        with status: st.write(f"✅ {protein_name[:80]}")
        progress.progress(20)

        center, size = get_binding_box(pdb_path)
        dock_score_a = dock_score_b = -7.0
        pose_atoms_a = pose_atoms_b = None
        docking_ran  = False

        if vina_cmd and obabel_cmd:
            # Receptor
            with status: st.write("⚙️ Preparing receptor...")
            receptor = prepare_receptor(pdb_path, work_dir, obabel_cmd)
            progress.progress(30)

            if receptor:
                with status: st.write("✅ Receptor ready")

                # Drug A
                with status: st.write(f"🔬 Docking {name_a or 'Drug A'}...")
                lig_a = prepare_ligand(smiles_a, "drug_a", work_dir, obabel_cmd)
                if lig_a:
                    out_a = os.path.join(work_dir, "drug_a_out.pdbqt")
                    score_a, stdout_a, stderr_a = run_vina(vina_cmd, receptor, lig_a, center, size, out_a, exhaustiveness)
                    if score_a is not None:
                        dock_score_a = score_a
                        pose_atoms_a = read_pose_atoms(out_a)
                        docking_ran  = True
                        with status: st.write(f"✅ {name_a or 'Drug A'}: {score_a:.2f} kcal/mol")
                    else:
                        with status: st.write(f"⚠️ Score parse failed")
                progress.progress(60)

                # Drug B
                with status: st.write(f"🔬 Docking {name_b or 'Drug B'}...")
                lig_b = prepare_ligand(smiles_b, "drug_b", work_dir, obabel_cmd)
                if lig_b:
                    out_b = os.path.join(work_dir, "drug_b_out.pdbqt")
                    score_b, stdout_b, stderr_b = run_vina(vina_cmd, receptor, lig_b, center, size, out_b, exhaustiveness)
                    if score_b is not None:
                        dock_score_b = score_b
                        pose_atoms_b = read_pose_atoms(out_b)
                        docking_ran  = True
                        with status: st.write(f"✅ {name_b or 'Drug B'}: {score_b:.2f} kcal/mol")
            else:
                with status: st.write("⚠️ Receptor prep failed")
        else:
            with status: st.write(f"⚠️ Tools missing: vina={vina_cmd} obabel={obabel_cmd}")

        progress.progress(75)

        # Synergy
        with status: st.write("🧠 Predicting synergy...")
        go_emb = torch.zeros(512).unsqueeze(0)
        dock   = torch.tensor([[float(dock_score_a), float(dock_score_b)]])
        with torch.no_grad():
            score, logit = model(Batch.from_data_list([ga]), Batch.from_data_list([gb]), go_emb, dock)
            synergy_score = score.item()
            synergy_prob  = torch.sigmoid(logit).item()
        progress.progress(100)
        with status: st.write("✅ Done!")

        # Visualization
        with viz_placeholder.container():
            if docking_ran and (pose_atoms_a or pose_atoms_b):
                st.markdown("**Both drugs docked in protein binding pocket**")
                show_docking_3d(pdb_content, pose_atoms_a, pose_atoms_b,
                                name_a or "Drug A", name_b or "Drug B")
                st.caption(f"🔵 {name_a or 'Drug A'} &nbsp; 🟠 {name_b or 'Drug B'} &nbsp; 🎨 Protein &nbsp; *Drag to rotate*")
            else:
                show_drugs_3d(smiles_a, smiles_b)
                st.caption("Docking poses unavailable — showing drug structures only")

        # Results
        st.markdown("---")
        st.markdown("### 📊 Results")
        if synergy_score > 4.0:    verdict, color = "✅ Strongly Synergistic", "green"
        elif synergy_score > 2.0:  verdict, color = "⚠️ Mildly Synergistic", "orange"
        elif synergy_score > -1.0: verdict, color = "➖ Approximately Additive", "blue"
        else:                      verdict, color = "❌ Antagonistic", "red"

        m1,m2,m3,m4 = st.columns(4)
        m1.metric("Synergy Score", f"{synergy_score:.3f}")
        m2.metric("Synergy Probability", f"{synergy_prob:.3f}")
        m3.metric(f"{name_a or 'Drug A'} Binding", f"{dock_score_a:.2f} kcal/mol")
        m4.metric(f"{name_b or 'Drug B'} Binding", f"{dock_score_b:.2f} kcal/mol")
        st.markdown(f"### Verdict: :{color}[{verdict}]")

        with st.expander("📋 Full report"):
            st.markdown(f"""
| Property | Value |
|----------|-------|
| Protein | {protein_name[:80]} |
| PDB ID | {pdb_id} |
| {name_a or 'Drug A'} docking | {dock_score_a:.3f} kcal/mol |
| {name_b or 'Drug B'} docking | {dock_score_b:.3f} kcal/mol |
| Synergy score | {synergy_score:.3f} |
| Verdict | {verdict} |
| Docking ran | {'Yes' if docking_ran else 'No'} |
| Box center | {[round(c,1) for c in center]} |
| Box size | {[round(s,1) for s in size]} Å |
            """)

        with st.expander("📖 How to interpret"):
            st.markdown("""
| Score | Meaning |
|-------|---------|
| > 4.0 | Strongly Synergistic |
| 2–4 | Mildly Synergistic |
| -1–2 | Approximately Additive |
| < -1 | Antagonistic |
            """)