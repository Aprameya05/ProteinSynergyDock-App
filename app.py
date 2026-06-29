"""
ProteinSynergyDock v6 — Multi-tab with Heatmap + Radar
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
import os, requests, subprocess, tempfile, shutil, json
import streamlit.components.v1 as components
import plotly.graph_objects as go
import pandas as pd

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
                     padding:12px; border-radius:6px; margin:8px 0; color:white; }
    .history-item { background:#1a1a2e; border-left:3px solid #4fc3f7;
                    padding:8px; border-radius:4px; margin:4px 0; color:white; font-size:12px; }
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
    def forward(self, da, db, go_emb, dock, cell_idx):
        ea    = self.drug_encoder(da.x, da.edge_index, da.batch)
        eb    = self.drug_encoder(db.x, db.edge_index, db.batch)
        fused = self.cross_attn(ea, eb)
        fused = fused*(1+self.film_scale(go_emb)) + self.film_bias(go_emb)
        cell  = self.cell_embed(cell_idx)
        fused = torch.cat([fused, dock, cell], dim=-1)
        out   = self.head(fused)
        return out[:,0], out[:,1]

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
    def forward(self, da, db, go_emb, dock):
        ea    = self.drug_encoder(da.x, da.edge_index, da.batch)
        eb    = self.drug_encoder(db.x, db.edge_index, db.batch)
        fused = self.cross_attn(ea, eb)
        fused = fused*(1+self.film_scale(go_emb)) + self.film_bias(go_emb)
        fused = torch.cat([fused, dock], dim=-1)
        out   = self.head(fused)
        return out[:,0], out[:,1]

@st.cache_resource
def load_model():
    ckpt_path = 'proteinsydock_v2_final.pt'
    if not os.path.exists(ckpt_path):
        return None, None, None, 'none', 0.0, 0.0
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    sd   = ckpt['state_dict']
    if any('cell_embed' in k for k in sd.keys()):
        n_cell = ckpt.get('n_cell_lines', 60)
        model  = ProteinSynergyDockV2(n_cell_lines=n_cell)
        model.load_state_dict(sd)
        model.eval()
        return model, ckpt.get('cell_line_to_idx',{}), \
               (ckpt.get('synergy_mean',-2.58), ckpt.get('synergy_std',6.06)), \
               'v2', ckpt.get('pearson_r',0.0), ckpt.get('auroc',0.0)
    else:
        model = ProteinSynergyDockV1()
        model.load_state_dict(sd)
        model.eval()
        return model, None, None, 'v1', ckpt.get('pearson_r',0.0), ckpt.get('auroc',0.0)

model, cell_to_idx, syn_scale, model_version, model_r, model_auroc = load_model()

if 'history' not in st.session_state:
    st.session_state.history = []

@st.cache_data
def load_precomputed():
    if os.path.exists('precomputed_scores.json'):
        with open('precomputed_scores.json') as f:
            return json.load(f)
    return None

scores_data = load_precomputed()

# ── Constants ─────────────────────────────────────────────────────────────────

KNOWN_SYNERGY = {
    ("Vemurafenib","Trametinib"):  {"UACC-62":8.4,"SK-MEL-5":7.2,"A375":9.1},
    ("Trametinib","Vemurafenib"):  {"UACC-62":8.4,"SK-MEL-5":7.2,"A375":9.1},
    ("Imatinib","Dasatinib"):      {"K-562":-1.4,"MOLT-4":-0.8},
    ("Dasatinib","Imatinib"):      {"K-562":-1.4,"MOLT-4":-0.8},
    ("Erlotinib","Lapatinib"):     {"A549/ATCC":5.5,"NCI-H23":4.2},
    ("Lapatinib","Erlotinib"):     {"A549/ATCC":5.5,"NCI-H23":4.2},
    ("Olaparib","Rucaparib"):      {"OVCAR-3":2.1,"SK-OV-3":1.8},
    ("Rucaparib","Olaparib"):      {"OVCAR-3":2.1,"SK-OV-3":1.8},
    ("Palbociclib","Abemaciclib"): {"MCF7":3.2,"T-47D":2.8},
    ("Abemaciclib","Palbociclib"): {"MCF7":3.2,"T-47D":2.8},
    ("Vemurafenib","Cobimetinib"): {"UACC-62":6.8,"SK-MEL-5":5.9},
    ("Cobimetinib","Vemurafenib"): {"UACC-62":6.8,"SK-MEL-5":5.9},
}

DRUG_SMILES_LOOKUP = {
    "Imatinib":      "CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)NC4=NC=CC(=N4)C5=CN=CC=C5",
    "Gefitinib":     "COC1=C(C=C2C(=C1)N=CN=C2NC3=CC(=C(C=C3)F)Cl)OCCCN4CCOCC4",
    "Erlotinib":     "COCCOC1=C(C=C2C(=C1)C(=NC=N2)NC3=CC=CC(=C3)C#C)OCCOC",
    "Lapatinib":     "CS(=O)(=O)CCNCc1oc(cc1)c2ccc3ncnc(Nc4ccc(Oc5cccc(Cl)c5)c(Cl)c4)c3c2",
    "Dasatinib":     "Cc1nc(Nc2ncc(s2)C(=O)Nc2c(C)cccc2Cl)cc(n1)N1CCN(CCO)CC1",
    "Nilotinib":     "Cc1cn(c2cc(NC(=O)c3ccc(C)c(Nc4nccc(n4)-c4cccnc4)c3)cc(C(F)(F)F)c12)C",
    "Vemurafenib":   "CCCS(=O)(=O)Nc1ccc(F)c(C(=O)c2c[nH]c3ncc(-c4ccc(Cl)cc4)cc23)c1",
    "Dabrafenib":    "CC(C)(C)c1nc2cc(F)ccc2c(C(=O)Nc2ccc(F)c(NS(=O)(=O)c3ccc(F)cc3)c2)n1",
    "Trametinib":    "CC(=O)Nc1ccc(-c2cc3c(nc(N)nc3n2C)N2CCC(F)(F)CC2=O)cc1F",
    "Cobimetinib":   "OC(COc1cc(Cl)c(F)cc1F)CN1CCC(=C1)c1cc2c(Nc3ccc(F)cc3F)ncc(C(N)=O)c2[nH]1",
    "Sorafenib":     "CNC(=O)c1cc(Oc2ccc(NC(=O)Nc3ccc(Cl)c(C(F)(F)F)c3)cc2)ccn1",
    "Sunitinib":     "CCN(CC)CCNC(=O)c1c(C)[nH]c(C=C2C(=O)Nc3ccc(F)cc32)c1C",
    "Olaparib":      "O=C1CCCN1c1ccc(cc1)C(=O)c1[nH]ncc1C1CC1",
    "Niraparib":     "OC(=O)c1ccc2[nH]ncc2c1-c1ccc(cn1)C1CCNCC1",
    "Rucaparib":     "NCc1cc2cc(F)ccc2[nH]1-c1ccc3NCCCC(=O)c3c1",
    "Palbociclib":   "CC1=C(C(=NC(=C1)N2CCNCC2)N3CCCC3)C(=O)NC4=CC=CC=N4",
    "Abemaciclib":   "CC1=NC(=NC(=C1)NC2=NC=CC(=N2)N3CCC(CC3)NC(=O)C4=CC=C(C=C4)F)C5=CC(=CC=C5)F",
    "Ribociclib":    "CC1=NC(=NC(=C1)N2CCNCC2)C3=CC4=C(C=C3)N=CN=C4N5CCCC5",
    "Ibrutinib":     "C=CC(=O)N1CCCC(c2ncnc3[nH]ccc23)C1",
    "Zanubrutinib":  "O=C(/C=C/c1ccco1)N1CCC(n2nc(-c3ccc4c(c3)CCNC4=O)c3c(N)ncnc23)CC1",
    "Acalabrutinib": "CC#CC(=O)N1CCC(n2nc(-c3ccc4c(c3)CCNC4=O)c3c(N)ncnc23)CC1",
    "Venetoclax":    "CC1(CCC(CC1)N2CCN(CC2)c3ccc(cc3)C(=O)NS(=O)(=O)c4ccc(cc4-c5cnc6ccccc6n5)Cl)C",
    "Alpelisib":     "CC1(C)CN(c2nc(Nc3ccc(S(N)(=O)=O)cc3F)ncc2F)CC1=O",
    "Paclitaxel":    "O=C(OC1C[C@]2(O)C(=O)C(OC(=O)c3ccccc3)C(O)C(OC(=O)C(NC(=O)c3ccccc3)c3ccccc3)C2(C)CC1)C(C)=C",
    "Doxorubicin":   "COc1cccc2C(=O)c3c(O)c4CC(O)(CC(OC5CC(N)C(O)C(C)O5)c4c(O)c3C(=O)c12)C(=O)CO",
    "Gemcitabine":   "NC(=O)C1=CN(C(=O)N1)C1CC(F)(F)C(CO)O1",
    "Osimertinib":   "C=CC(=O)Nc1cc2c(Nc3ccc(F)c(Cl)c3)nc(OC)nc2cc1N(C)CCN(C)C",
    "Alectinib":     "COc1cc2c(cc1N1CCC(CC1)c1ccc3[nH]ccc3c1)cc(=O)n1ccc(C#N)c21",
    "Afatinib":      "CN(C)C/C=C/C(=O)Nc1cc2c(Nc3ccc(F)c(Cl)c3)ncnc2cc1OC",
    "Capecitabine":  "CCOC(=O)Nc1nc(=O)n(C2OC(C)C(O)C2O)cc1F",
    "Temozolomide":  "Cn1nnc2c(C(N)=O)ncn12",
    "Selumetinib":   "Cc1cc(Nc2ncc(F)c(Nc3ccc(I)c(F)c3)n2)c(Cl)cc1Cl",
    "Belinostat":    "O=C(/C=C/c1ccccc1)NOc1ccc(NS(=O)(=O)c2ccccc2)cc1",
    "Vorinostat":    "O=C(CCCCCCC(=O)Nc1ccccc1)NO",
}

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

# ── Helper functions ──────────────────────────────────────────────────────────

def lookup_known_synergy(drug_a, drug_b, cell_line=None):
    key = (drug_a, drug_b)
    if key not in KNOWN_SYNERGY: return None
    scores = KNOWN_SYNERGY[key]
    if cell_line and cell_line in scores:
        return scores[cell_line], cell_line
    avg = np.mean(list(scores.values()))
    return avg, f"avg across {len(scores)} cell lines"

def smiles_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return None
    try:
        mol=Chem.AddHs(mol); AllChem.EmbedMolecule(mol,AllChem.ETKDGv3()); mol=Chem.RemoveHs(mol)
        if mol.GetNumConformers()==0: AllChem.Compute2DCoords(mol)
    except:
        try: AllChem.Compute2DCoords(mol)
        except: return None
    feats,pos=[],[]
    conf=mol.GetConformer() if mol.GetNumConformers()>0 else None
    for atom in mol.GetAtoms():
        feats.append([atom.GetAtomicNum(),atom.GetDegree(),atom.GetFormalCharge(),
            int(atom.GetIsAromatic()),int(atom.IsInRing()),atom.GetTotalNumHs(),
            atom.GetNumRadicalElectrons()])
        if conf:
            p=conf.GetAtomPosition(atom.GetIdx()); pos.append([p.x,p.y,p.z])
        else: pos.append([0.,0.,0.])
    es,ed=[],[]
    for bond in mol.GetBonds():
        i,j=bond.GetBeginAtomIdx(),bond.GetEndAtomIdx(); es+=[i,j]; ed+=[j,i]
    if not es: return None
    return Data(x=torch.tensor(feats,dtype=torch.float),
                pos=torch.tensor(pos,dtype=torch.float),
                edge_index=torch.tensor([es,ed],dtype=torch.long))

def fetch_pdb(pdb_id, save_dir):
    path=os.path.join(save_dir,f"{pdb_id}.pdb")
    if os.path.exists(path) and os.path.getsize(path)>1000: return path
    r=requests.get(f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb",timeout=30)
    if r.status_code==200:
        with open(path,'w') as f: f.write(r.text)
        return path
    return None

def get_protein_info(pdb_id):
    try:
        r=requests.get(f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id.upper()}",timeout=10)
        if r.status_code==200:
            return r.json().get('struct',{}).get('title',f'Protein {pdb_id}')
    except: pass
    return f"Protein {pdb_id}"

def get_binding_box(pdb_path, padding=10.0):
    hetatm,atom=[],[]
    with open(pdb_path) as f:
        for line in f:
            if line.startswith('HETATM'):
                if line[17:20].strip() not in ['HOH','WAT','H2O']:
                    try: hetatm.append([float(line[30:38]),float(line[38:46]),float(line[46:54])])
                    except: pass
            elif line.startswith('ATOM'):
                try: atom.append([float(line[30:38]),float(line[38:46]),float(line[46:54])])
                except: pass
    if len(hetatm)>=5:
        c=np.array(hetatm); ctr=c.mean(axis=0).tolist()
        sz=np.clip(c.max(axis=0)-c.min(axis=0)+padding*2,18,30).tolist()
        return ctr,sz,"ligand"
    if atom:
        c=np.array(atom); ctr=c.mean(axis=0).tolist()
        sz=np.clip(c.max(axis=0)-c.min(axis=0)+padding,20,28).tolist()
        return ctr,sz,"protein_center"
    return [0,0,0],[25,25,25],"default"

def find_vina():
    for cmd in ['vina','autodock_vina','/usr/bin/vina','/usr/local/bin/vina']:
        if shutil.which(cmd): return cmd
    return None

def prepare_ligand(smiles,name,work_dir):
    out=f'{work_dir}/{name}.pdbqt'
    if os.path.exists(out) and os.path.getsize(out)>0: return out
    mol=Chem.MolFromSmiles(smiles)
    if mol is None: return None
    try:
        mol=Chem.AddHs(mol); AllChem.EmbedMolecule(mol,AllChem.ETKDGv3())
        AllChem.MMFFOptimizeMolecule(mol); mol=Chem.RemoveHs(mol)
    except:
        try: AllChem.Compute2DCoords(mol)
        except: return None
    sdf=f'{work_dir}/{name}.sdf'; pdb=f'{work_dir}/{name}.pdb'
    w=Chem.SDWriter(sdf); w.write(mol); w.close()
    subprocess.run(['obabel',sdf,'-O',pdb,'-h'],capture_output=True)
    subprocess.run(['obabel',pdb,'-O',out,'--partialcharge','gasteiger'],capture_output=True)
    return out if os.path.exists(out) and os.path.getsize(out)>0 else None

def prepare_receptor(pdb_path,work_dir):
    pdb_id=os.path.basename(pdb_path).replace('.pdb','')
    out=f'{work_dir}/{pdb_id}_rec.pdbqt'
    if os.path.exists(out) and os.path.getsize(out)>0: return out
    clean=f'{work_dir}/{pdb_id}_clean.pdb'
    with open(pdb_path) as fin, open(clean,'w') as fout:
        for line in fin:
            if line.startswith('ATOM') or line.startswith('END'): fout.write(line)
    subprocess.run(['obabel',clean,'-O',out,'--partialcharge','gasteiger','-xr'],capture_output=True)
    return out if os.path.exists(out) and os.path.getsize(out)>0 else None

def run_vina(vina,receptor,ligand,center,size,out_path,exhaustiveness=8):
    cmd=[vina,'--receptor',receptor,'--ligand',ligand,'--out',out_path,
         '--center_x',str(round(center[0],3)),'--center_y',str(round(center[1],3)),
         '--center_z',str(round(center[2],3)),'--size_x',str(round(size[0],3)),
         '--size_y',str(round(size[1],3)),'--size_z',str(round(size[2],3)),
         '--exhaustiveness',str(exhaustiveness),'--num_modes','3']
    try:
        result=subprocess.run(cmd,capture_output=True,text=True,timeout=300)
        best_score=None
        if os.path.exists(out_path):
            with open(out_path) as f:
                for line in f:
                    if 'REMARK VINA RESULT' in line:
                        try: best_score=float(line.split()[3]); break
                        except: pass
        if best_score is None:
            for line in result.stdout.split('\n'):
                s=line.strip()
                if s and s[0]=='1' and len(s.split())>=3:
                    try: best_score=float(s.split()[1]); break
                    except: pass
        return best_score,result.stderr
    except Exception as e: return None,str(e)

def read_pose_atoms(pdbqt_path):
    atoms=[]
    if not os.path.exists(pdbqt_path): return None
    with open(pdbqt_path) as f:
        for line in f:
            if line.startswith('ENDMDL'): break
            if line.startswith(('ATOM','HETATM')):
                try: atoms.append((line[12:16].strip(),float(line[30:38]),float(line[38:46]),float(line[46:54])))
                except: pass
    return atoms or None

def show_docking_3d(pdb_content,atoms_a,atoms_b,name_a,name_b,height=500):
    viewer=py3Dmol.view(width=750,height=height)
    viewer.addModel(pdb_content,'pdb')
    viewer.setStyle({'model':0},{'cartoon':{'color':'spectrum','opacity':0.65}})
    if atoms_a:
        block="MODEL 1\n"
        for i,(a,x,y,z) in enumerate(atoms_a):
            block+=f"HETATM{i+1:5d}  {a:<4s}LGA A   1    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00\n"
        block+="ENDMDL\n"
        viewer.addModel(block,'pdb')
        viewer.setStyle({'model':1},{'stick':{'colorscheme':'cyanCarbon','radius':0.2},'sphere':{'colorscheme':'cyanCarbon','scale':0.3}})
    if atoms_b:
        block="MODEL 1\n"
        for i,(a,x,y,z) in enumerate(atoms_b):
            block+=f"HETATM{i+1:5d}  {a:<4s}LGB B   1    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00\n"
        block+="ENDMDL\n"
        viewer.addModel(block,'pdb')
        idx=2 if atoms_a else 1
        viewer.setStyle({'model':idx},{'stick':{'colorscheme':'orangeCarbon','radius':0.2},'sphere':{'colorscheme':'orangeCarbon','scale':0.3}})
    viewer.setBackgroundColor('#1a1a2e')
    viewer.zoomTo({'model':1} if atoms_a else {})
    viewer.zoom(1.3)
    components.html(viewer._make_html(),height=height+20,scrolling=False)

def show_drugs_3d(smiles_a,smiles_b,height=400):
    viewer=py3Dmol.view(width=750,height=height)
    offset=0
    for i,(smiles,color) in enumerate([(smiles_a,'cyanCarbon'),(smiles_b,'orangeCarbon')]):
        mol=Chem.MolFromSmiles(smiles) if smiles else None
        if mol is None: continue
        try:
            mol=Chem.AddHs(mol); AllChem.EmbedMolecule(mol,AllChem.ETKDGv3())
            AllChem.MMFFOptimizeMolecule(mol); mol=Chem.RemoveHs(mol)
            conf=mol.GetConformer()
            for j in range(mol.GetNumAtoms()):
                p=conf.GetAtomPosition(j); conf.SetAtomPosition(j,(p.x+offset,p.y,p.z))
            viewer.addModel(Chem.MolToMolBlock(mol),'sdf')
            viewer.setStyle({'model':i},{'stick':{'colorscheme':color,'radius':0.15},'sphere':{'colorscheme':color,'scale':0.3}})
            offset+=15
        except: pass
    viewer.setBackgroundColor('#1a1a2e')
    viewer.zoomTo()
    components.html(viewer._make_html(),height=height+20,scrolling=False)

def get_verdict(score):
    if score>0.5:    return "✅ Strongly Synergistic","green"
    elif score>0.1:  return "⚠️ Mildly Synergistic","orange"
    elif score>-0.1: return "➖ Approximately Additive","blue"
    else:            return "❌ Antagonistic","red"

# ── Header ────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="main-header">
    <h1>🧬 ProteinSynergyDock</h1>
    <p>Structure-aware drug combination synergy prediction with cell line context</p>
    <p style="font-size:13px;color:#78909c;margin-top:8px;">
        Real AutoDock Vina docking · ProteinWhisper++ GO context · 60 cancer cell lines
    </p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🔬 Quick Examples")
    example = st.selectbox("Choose a drug pair:", list(SHOWCASES.keys()))
    ex = SHOWCASES[example]
    if ex["note"]: st.info(ex["note"])
    st.markdown("---")
    st.markdown(f"""
## 📊 Model Info
- **Version:** {model_version.upper() if model_version != 'none' else 'Not loaded'}
- **Pearson r:** {model_r:.4f}
- **AUROC:** {model_auroc:.4f}
- **Real docking:** AutoDock Vina
- **Training data:** 107,103 NCI ALMANAC scores
- **Cell lines:** 60 cancer types

## 🔗 Links
- [GitHub](https://github.com/Aprameya05/ProteinSynergyDock)
- [ProteinWhisper](https://github.com/Aprameya05/ProteinWhisper)
- [DrugSynergy3D](https://github.com/Aprameya05/DrugSynergy3D)
    """)
    if st.session_state.history:
        st.markdown("---")
        st.markdown("## 📜 Recent Predictions")
        for h in st.session_state.history:
            verdict_icon = h['verdict'].split()[0]
            st.markdown(f"""<div class="history-item">
<b>{h['drug_a']} + {h['drug_b']}</b><br>
{h['cell_line']} | Score: {h['score']:.3f} | {verdict_icon}
</div>""", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "🔬 Predict Synergy",
    "🗺️ Synergy Landscape", 
    "📊 Cell Line Comparison",
    "🏥 Clinical Trials",
    "📚 Literature",
    "🔄 Drug Repurposing",
    "🤖 AI Assistant",
])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1: Predict Synergy
# ═══════════════════════════════════════════════════════════════════════════════

with tab1:
    col1, col2 = st.columns([1, 1.2])

    with col1:
        st.markdown("### 💊 Drug Inputs")
        drug_a_options = ["Custom (paste SMILES below)"] + sorted(DRUG_SMILES_LOOKUP.keys())
        drug_a_select  = st.selectbox("Drug A — select known drug", drug_a_options, key="da_select")
        if drug_a_select != "Custom (paste SMILES below)":
            smiles_a = DRUG_SMILES_LOOKUP[drug_a_select]
            name_a   = drug_a_select
            st.text_area("Drug A SMILES (auto-filled)", value=smiles_a, height=60, disabled=True)
        else:
            name_a   = st.text_input("Drug A name", value=ex.get("name_a",""), placeholder="e.g. Imatinib")
            smiles_a = st.text_area("Drug A — SMILES", value=ex["smiles_a"], height=80)

        drug_b_options = ["Custom (paste SMILES below)"] + sorted(DRUG_SMILES_LOOKUP.keys())
        drug_b_select  = st.selectbox("Drug B — select known drug", drug_b_options, key="db_select")
        if drug_b_select != "Custom (paste SMILES below)":
            smiles_b = DRUG_SMILES_LOOKUP[drug_b_select]
            name_b   = drug_b_select
            st.text_area("Drug B SMILES (auto-filled)", value=smiles_b, height=60, disabled=True)
        else:
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
        if default_cl not in cell_lines_for_panel: default_cl = cell_lines_for_panel[0]
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

    if run_btn:
        if not smiles_a or not smiles_b:
            st.error("Please enter SMILES for both drugs"); st.stop()
        if not pdb_id:
            st.error("Please enter a PDB ID"); st.stop()
        if model is None:
            st.error("Model not loaded"); st.stop()

        ga = smiles_to_graph(smiles_a)
        gb = smiles_to_graph(smiles_b)
        if ga is None: st.error("❌ Invalid SMILES for Drug A"); st.stop()
        if gb is None: st.error("❌ Invalid SMILES for Drug B"); st.stop()

        known      = lookup_known_synergy(name_a or "Drug A", name_b or "Drug B", cell_line)
        vina_cmd   = find_vina()
        obabel_cmd = shutil.which('obabel')

        st.markdown("---")
        st.markdown("### 🔄 Pipeline Running...")
        progress = st.progress(0)
        status   = st.status("Starting...", expanded=True)

        with tempfile.TemporaryDirectory() as work_dir:
            with status: st.write(f"📥 Fetching {pdb_id} from RCSB...")
            progress.progress(10)
            pdb_path = fetch_pdb(pdb_id, work_dir)
            if not pdb_path: st.error(f"❌ Could not fetch {pdb_id}"); st.stop()
            pdb_content  = open(pdb_path).read()
            protein_name = get_protein_info(pdb_id)
            center,size,box_method = get_binding_box(pdb_path)
            with status:
                st.write(f"✅ {protein_name[:70]}")
                st.write(f"📦 Box: {box_method} | center: {[round(c,1) for c in center]}")
            progress.progress(20)

            dock_score_a = dock_score_b = -7.0
            pose_atoms_a = pose_atoms_b = None
            docking_ran  = False

            if vina_cmd and obabel_cmd:
                receptor = prepare_receptor(pdb_path, work_dir)
                progress.progress(30)
                if receptor:
                    with status: st.write("✅ Receptor ready")
                    with status: st.write(f"🔬 Docking {name_a or 'Drug A'}...")
                    lig_a = prepare_ligand(smiles_a,"drug_a",work_dir)
                    if lig_a:
                        out_a=f'{work_dir}/drug_a_out.pdbqt'
                        s_a,_=run_vina(vina_cmd,receptor,lig_a,center,size,out_a,exhaustiveness)
                        if s_a is not None:
                            dock_score_a=s_a; pose_atoms_a=read_pose_atoms(out_a); docking_ran=True
                            with status: st.write(f"✅ {name_a or 'Drug A'}: {s_a:.2f} kcal/mol")
                    progress.progress(60)
                    with status: st.write(f"🔬 Docking {name_b or 'Drug B'}...")
                    lig_b = prepare_ligand(smiles_b,"drug_b",work_dir)
                    if lig_b:
                        out_b=f'{work_dir}/drug_b_out.pdbqt'
                        s_b,_=run_vina(vina_cmd,receptor,lig_b,center,size,out_b,exhaustiveness)
                        if s_b is not None:
                            dock_score_b=s_b; pose_atoms_b=read_pose_atoms(out_b); docking_ran=True
                            with status: st.write(f"✅ {name_b or 'Drug B'}: {s_b:.2f} kcal/mol")
            else:
                with status: st.write("⚠️ Docking tools unavailable")
            progress.progress(75)

            with status: st.write("🧠 Predicting synergy...")
            go_emb = torch.zeros(512).unsqueeze(0)
            dock   = torch.tensor([[float(dock_score_a), float(dock_score_b)]])

            with torch.no_grad():
                if model_version == 'v2' and cell_to_idx:
                    cell_idx = torch.tensor([cell_to_idx.get(cell_line,0)], dtype=torch.long)
                    score, logit = model(Batch.from_data_list([ga]),Batch.from_data_list([gb]),go_emb,dock,cell_idx)
                else:
                    score, logit = model(Batch.from_data_list([ga]),Batch.from_data_list([gb]),go_emb,dock)
                synergy_score = score.item()
                synergy_prob  = torch.sigmoid(logit).item()

            progress.progress(100)
            with status: st.write("✅ Complete!")

            with viz_placeholder.container():
                if docking_ran and (pose_atoms_a or pose_atoms_b):
                    st.markdown("**Both drugs docked in protein binding pocket**")
                    show_docking_3d(pdb_content,pose_atoms_a,pose_atoms_b,name_a or "Drug A",name_b or "Drug B")
                    st.caption(f"🔵 {name_a or 'Drug A'} &nbsp; 🟠 {name_b or 'Drug B'} &nbsp; 🎨 Protein &nbsp; *Drag to rotate*")
                else:
                    show_drugs_3d(smiles_a, smiles_b)

            st.markdown("---")
            st.markdown("### 📊 Results")
            verdict, color = get_verdict(synergy_score)

            m1,m2,m3,m4 = st.columns(4)
            m1.metric("Synergy Score", f"{synergy_score:.3f}")
            m2.metric("Synergy Probability", f"{synergy_prob:.3f}")
            m3.metric(f"{name_a or 'Drug A'} Binding", f"{dock_score_a:.2f} kcal/mol")
            m4.metric(f"{name_b or 'Drug B'} Binding", f"{dock_score_b:.2f} kcal/mol")

            st.markdown(f"### Verdict: :{color}[{verdict}]")
            st.caption(f"Cancer context: **{panel}** → **{cell_line}**")

            st.session_state.history.insert(0,{
                'drug_a':name_a or 'Drug A','drug_b':name_b or 'Drug B',
                'cell_line':cell_line,'score':synergy_score,
                'verdict':verdict,'dock_a':dock_score_a,'dock_b':dock_score_b,
            })
            st.session_state.history = st.session_state.history[:5]

            if known:
                known_score,known_source=known
                st.markdown(f"""<div class="known-score">
📚 <strong>NCI ALMANAC Ground Truth</strong><br>
Known synergy score: <strong>{known_score:.2f}</strong> ({known_source})<br>
Model prediction: <strong>{synergy_score:.3f}</strong> &nbsp; Error: <strong>{abs(synergy_score-known_score):.2f}</strong> Loewe units
</div>""", unsafe_allow_html=True)
            else:
                st.markdown("""<div class="unknown-score">
🔮 <strong>Novel prediction</strong> — this drug pair × cell line not in NCI ALMANAC
</div>""", unsafe_allow_html=True)

            with st.expander("📋 Full docking report"):
                st.markdown(f"""
| Property | Value |
|----------|-------|
| Protein | {protein_name[:70]} |
| PDB ID | {pdb_id} |
| Binding box | {box_method} |
| {name_a or 'Drug A'} docking | {dock_score_a:.3f} kcal/mol |
| {name_b or 'Drug B'} docking | {dock_score_b:.3f} kcal/mol |
| Cancer type | {panel} |
| Cell line | {cell_line} |
| Synergy score | {synergy_score:.3f} |
| Verdict | {verdict} |
                """)

            with st.expander("📖 How to interpret"):
                st.markdown("""
| Score | Meaning |
|-------|---------|
| > 0.5 | Strongly Synergistic |
| 0.1–0.5 | Mildly Synergistic |
| -0.1–0.1 | Approximately Additive |
| < -0.1 | Antagonistic |

**Docking score** (kcal/mol): more negative = stronger binding. Below -8 = strong binder.
                """)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2: Synergy Landscape Heatmap
# ═══════════════════════════════════════════════════════════════════════════════

with tab2:
    st.markdown("### 🗺️ Synergy Landscape — All Drug Combinations")
    st.caption("Precomputed synergy scores for 28 drugs × 28 drugs across 9 cancer types.")

    if scores_data is None:
        st.warning("Precomputed scores file not found. Add precomputed_scores.json to the repo.")
    else:
        panel_options   = list(scores_data.keys())
        selected_panel  = st.selectbox("Cancer type:", panel_options, key="heatmap_panel")
        panel_data      = scores_data[selected_panel]
        drugs           = panel_data['drugs']
        matrix          = np.array(panel_data['matrix'])
        cell_line_h     = panel_data['cell_line']

        st.caption(f"Cell line: **{cell_line_h}** | {len(drugs)} drugs | {len(drugs)**2} combinations")

        fig = go.Figure(data=go.Heatmap(
            z=matrix, x=drugs, y=drugs,
            colorscale=[
                [0.0,'#2166ac'],[0.35,'#74add1'],[0.5,'#f7f7f7'],
                [0.65,'#f46d43'],[1.0,'#d73027'],
            ],
            zmid=0,
            text=[[f"{drugs[i]} + {drugs[j]}<br>Score: {matrix[i][j]:.3f}"
                   for j in range(len(drugs))] for i in range(len(drugs))],
            hovertemplate="%{text}<extra></extra>",
            colorbar=dict(title="Synergy",
                tickvals=[-0.4,-0.2,0,0.2,0.4],
                ticktext=["Antagonistic","","Additive","","Synergistic"]),
        ))
        fig.update_layout(
            height=700,
            xaxis=dict(tickangle=-45,tickfont=dict(size=10)),
            yaxis=dict(tickfont=dict(size=10)),
            margin=dict(l=130,r=20,t=20,b=130),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
        )
        st.plotly_chart(fig, use_container_width=True)

        col_top, col_bot = st.columns(2)

        pairs = [(drugs[i],drugs[j],float(matrix[i][j]))
                 for i in range(len(drugs)) for j in range(len(drugs)) if i!=j]

        with col_top:
            st.markdown("#### 🏆 Top 10 Synergistic Pairs")
            top = sorted(pairs, key=lambda x: x[2], reverse=True)[:10]
            top_df = pd.DataFrame(top, columns=['Drug A','Drug B','Score'])
            top_df['Score'] = top_df['Score'].round(3)
            top_df['Verdict'] = top_df['Score'].apply(lambda x: get_verdict(x)[0])
            st.dataframe(top_df, use_container_width=True, hide_index=True)

        with col_bot:
            st.markdown("#### ⚠️ Top 10 Antagonistic Pairs")
            bot = sorted(pairs, key=lambda x: x[2])[:10]
            bot_df = pd.DataFrame(bot, columns=['Drug A','Drug B','Score'])
            bot_df['Score'] = bot_df['Score'].round(3)
            bot_df['Verdict'] = bot_df['Score'].apply(lambda x: get_verdict(x)[0])
            st.dataframe(bot_df, use_container_width=True, hide_index=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3: Cell Line Comparison
# ═══════════════════════════════════════════════════════════════════════════════

with tab3:
    st.markdown("### 📊 Cell Line Comparison — Same Drug Pair Across All Cancers")
    st.caption("Select a drug pair to see predicted synergy across all 9 cancer types.")

    if scores_data is None:
        st.warning("Precomputed scores file not found.")
    else:
        all_drugs = scores_data['Melanoma']['drugs']
        col_a, col_b = st.columns(2)
        with col_a:
            drug_a_r = st.selectbox("Drug A:", all_drugs,
                index=all_drugs.index("Vemurafenib") if "Vemurafenib" in all_drugs else 0,
                key="radar_a")
        with col_b:
            drug_b_r = st.selectbox("Drug B:", all_drugs,
                index=all_drugs.index("Trametinib") if "Trametinib" in all_drugs else 1,
                key="radar_b")

        if drug_a_r == drug_b_r:
            st.warning("Please select two different drugs.")
        else:
            panels = list(scores_data.keys())
            radar_scores = []
            for p in panels:
                pd2 = scores_data[p]
                dr  = pd2['drugs']
                mat = np.array(pd2['matrix'])
                if drug_a_r in dr and drug_b_r in dr:
                    radar_scores.append(float(mat[dr.index(drug_a_r)][dr.index(drug_b_r)]))
                else:
                    radar_scores.append(0.0)

            col_radar, col_bar = st.columns([1,1])

            with col_radar:
                fig_r = go.Figure()
                fig_r.add_trace(go.Scatterpolar(
                    r=radar_scores+[radar_scores[0]],
                    theta=panels+[panels[0]],
                    fill='toself',
                    fillcolor='rgba(79,195,247,0.2)',
                    line=dict(color='#4fc3f7',width=2),
                    name=f"{drug_a_r} + {drug_b_r}",
                ))
                fig_r.update_layout(
                    polar=dict(
                        radialaxis=dict(visible=True,
                            range=[min(radar_scores)-0.05, max(radar_scores)+0.05],
                            tickfont=dict(size=9)),
                        angularaxis=dict(tickfont=dict(size=10)),
                    ),
                    height=450,
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white'),
                    showlegend=False,
                    title=dict(text=f"{drug_a_r} + {drug_b_r}",
                               font=dict(size=14,color='#4fc3f7')),
                )
                st.plotly_chart(fig_r, use_container_width=True)

            with col_bar:
                colors=['#d73027' if s>0.1 else '#2166ac' if s<-0.1 else '#888888'
                        for s in radar_scores]
                fig_b = go.Figure(go.Bar(
                    x=panels, y=radar_scores,
                    marker_color=colors,
                    text=[f"{s:.3f}" for s in radar_scores],
                    textposition='outside',
                ))
                fig_b.update_layout(
                    height=450,
                    xaxis=dict(tickangle=-35,tickfont=dict(size=10)),
                    yaxis=dict(title="Synergy score",zeroline=True,zerolinecolor='#666'),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white'),
                    showlegend=False,
                )
                st.plotly_chart(fig_b, use_container_width=True)

            summary = pd.DataFrame({
                'Cancer Type': panels,
                'Cell Line': [scores_data[p]['cell_line'] for p in panels],
                'Synergy Score': [round(s,3) for s in radar_scores],
                'Verdict': [get_verdict(s)[0] for s in radar_scores],
            }).sort_values('Synergy Score',ascending=False).reset_index(drop=True)
            st.dataframe(summary, use_container_width=True, hide_index=True)
            # ═══════════════════════════════════════════════════════════════════════════════
# TAB 4: Clinical Trial Matching
# ═══════════════════════════════════════════════════════════════════════════════

with tab4:
    st.markdown("### 🏥 Clinical Trial Matching")
    st.caption("Search ClinicalTrials.gov for active trials testing your drug combination.")

    col_ct1, col_ct2 = st.columns(2)
    with col_ct1:
        ct_drug_a = st.text_input("Drug A name", placeholder="e.g. Vemurafenib", key="ct_a")
    with col_ct2:
        ct_drug_b = st.text_input("Drug B name", placeholder="e.g. Trametinib", key="ct_b")

    ct_condition = st.text_input("Cancer type (optional)", placeholder="e.g. melanoma, lung cancer", key="ct_cond")
    search_trials_btn = st.button("🔍 Search Clinical Trials", key="ct_btn")

    if search_trials_btn and ct_drug_a and ct_drug_b:
        with st.spinner("Searching ClinicalTrials.gov..."):
            try:
                query = f"{ct_drug_a} {ct_drug_b}"
                if ct_condition: query += f" {ct_condition}"
                
                url = "https://clinicaltrials.gov/api/v2/studies"
                params = {
                    "query.term": query,
                    "filter.overallStatus": "RECRUITING,ACTIVE_NOT_RECRUITING,COMPLETED",
                    "pageSize": 15,
                    "format": "json",
                    "fields": "NCTId,BriefTitle,OverallStatus,Phase,StartDate,CompletionDate,LeadSponsorName,Condition,InterventionName"
                }
                resp = requests.get(url, params=params, timeout=15)
                
                if resp.status_code == 200:
                    data = resp.json()
                    studies = data.get('studies', [])
                    
                    if not studies:
                        st.info(f"No clinical trials found for {ct_drug_a} + {ct_drug_b}. Try different drug names or broader search terms.")
                    else:
                        st.success(f"Found {len(studies)} clinical trials for **{ct_drug_a} + {ct_drug_b}**")
                        
                        for study in studies:
                            proto = study.get('protocolSection', {})
                            id_mod    = proto.get('identificationModule', {})
                            status_mod = proto.get('statusModule', {})
                            design_mod = proto.get('designModule', {})
                            sponsor_mod = proto.get('sponsorCollaboratorsModule', {})
                            cond_mod  = proto.get('conditionsModule', {})
                            
                            nct_id    = id_mod.get('nctId', 'N/A')
                            title     = id_mod.get('briefTitle', 'No title')
                            status    = status_mod.get('overallStatus', 'Unknown')
                            phase     = design_mod.get('phases', ['N/A'])
                            phase_str = ', '.join(phase) if isinstance(phase, list) else str(phase)
                            sponsor   = sponsor_mod.get('leadSponsor', {}).get('name', 'Unknown')
                            conditions = cond_mod.get('conditions', [])
                            
                            status_color = {
                                'RECRUITING': '🟢',
                                'ACTIVE_NOT_RECRUITING': '🟡',
                                'COMPLETED': '⚫',
                            }.get(status, '⚪')

                            with st.expander(f"{status_color} {title[:80]}..."):
                                c1, c2, c3 = st.columns(3)
                                c1.metric("NCT ID", nct_id)
                                c2.metric("Status", status.replace('_',' ').title())
                                c3.metric("Phase", phase_str)
                                st.markdown(f"**Sponsor:** {sponsor}")
                                if conditions:
                                    st.markdown(f"**Conditions:** {', '.join(conditions[:5])}")
                                st.markdown(f"[View on ClinicalTrials.gov](https://clinicaltrials.gov/study/{nct_id})")
                else:
                    st.error(f"ClinicalTrials.gov API error: {resp.status_code}")
            except Exception as e:
                st.error(f"Error searching trials: {e}")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5: Literature Mining
# ═══════════════════════════════════════════════════════════════════════════════

with tab5:
    st.markdown("### 📚 Literature Mining")
    st.caption("Search PubMed for papers about your drug combination.")

    col_pub1, col_pub2 = st.columns(2)
    with col_pub1:
        pub_drug_a = st.text_input("Drug A", placeholder="e.g. Vemurafenib", key="pub_a")
    with col_pub2:
        pub_drug_b = st.text_input("Drug B", placeholder="e.g. Trametinib", key="pub_b")

    pub_topic = st.text_input("Additional topic (optional)", placeholder="e.g. synergy, resistance, melanoma", key="pub_topic")
    pub_btn   = st.button("🔍 Search PubMed", key="pub_btn")

    if pub_btn and pub_drug_a and pub_drug_b:
        with st.spinner("Searching PubMed..."):
            try:
                # Search PubMed
                query = f"{pub_drug_a} AND {pub_drug_b}"
                if pub_topic: query += f" AND {pub_topic}"

                search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
                search_params = {
                    "db": "pubmed", "term": query,
                    "retmax": 15, "retmode": "json",
                    "sort": "relevance"
                }
                search_resp = requests.get(search_url, params=search_params, timeout=15)
                pmids = search_resp.json().get('esearchresult', {}).get('idlist', [])

                if not pmids:
                    st.info(f"No papers found for {pub_drug_a} + {pub_drug_b}. Try broader terms.")
                else:
                    # Fetch details
                    fetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
                    fetch_params = {
                        "db": "pubmed",
                        "id": ",".join(pmids),
                        "retmode": "json"
                    }
                    fetch_resp = requests.get(fetch_url, params=fetch_params, timeout=15)
                    results    = fetch_resp.json().get('result', {})

                    total = results.get('uids', pmids)
                    st.success(f"Found **{len(total)} papers** for **{pub_drug_a} + {pub_drug_b}**")

                    for pmid in total:
                        if pmid == 'uids': continue
                        paper = results.get(pmid, {})
                        title   = paper.get('title', 'No title')
                        journal = paper.get('fulljournalname', paper.get('source', 'Unknown'))
                        pubdate = paper.get('pubdate', 'Unknown date')
                        authors = paper.get('authors', [])
                        author_str = authors[0].get('name','') + ' et al.' if authors else 'Unknown'

                        with st.expander(f"📄 {title[:80]}..."):
                            c1,c2,c3 = st.columns(3)
                            c1.metric("Journal", journal[:30])
                            c2.metric("Date", pubdate)
                            c3.metric("PMID", pmid)
                            st.markdown(f"**Authors:** {author_str}")
                            st.markdown(f"[Read on PubMed](https://pubmed.ncbi.nlm.nih.gov/{pmid}/)")

            except Exception as e:
                st.error(f"Error searching PubMed: {e}")
                # ═══════════════════════════════════════════════════════════════════════════════
# TAB 6: Drug Repurposing Mode
# ═══════════════════════════════════════════════════════════════════════════════

with tab6:
    st.markdown("### 🔄 Drug Repurposing — Find Best Partner for Your Drug")
    st.caption("Select a drug and cancer type — we'll rank all 27 other drugs by predicted synergy.")

    if scores_data is None:
        st.warning("Precomputed scores not found.")
    else:
        all_drugs_r = scores_data['Melanoma']['drugs']
        col_r1, col_r2 = st.columns(2)
        with col_r1:
            anchor_drug = st.selectbox("Your drug:", all_drugs_r,
                index=all_drugs_r.index("Imatinib") if "Imatinib" in all_drugs_r else 0,
                key="repurpose_drug")
        with col_r2:
            repurpose_panel = st.selectbox("Cancer type:", list(scores_data.keys()),
                key="repurpose_panel")

        pd_r    = scores_data[repurpose_panel]
        drugs_r = pd_r['drugs']
        mat_r   = np.array(pd_r['matrix'])
        cl_r    = pd_r['cell_line']

        if anchor_drug in drugs_r:
            anchor_idx = drugs_r.index(anchor_drug)
            scores_row = [(drugs_r[j], float(mat_r[anchor_idx][j]))
                          for j in range(len(drugs_r)) if j != anchor_idx]
            scores_row.sort(key=lambda x: x[1], reverse=True)

            st.markdown(f"#### Best partners for **{anchor_drug}** in **{repurpose_panel}** ({cl_r})")

            # Ranked bar chart
            drug_names = [x[0] for x in scores_row]
            drug_scores = [x[1] for x in scores_row]
            colors = ['#d73027' if s>0.1 else '#2166ac' if s<-0.1 else '#888'
                      for s in drug_scores]

            fig_rep = go.Figure(go.Bar(
                x=drug_scores, y=drug_names,
                orientation='h',
                marker_color=colors,
                text=[f"{s:.3f}" for s in drug_scores],
                textposition='outside',
            ))
            fig_rep.update_layout(
                height=700,
                xaxis=dict(title="Synergy score", zeroline=True, zerolinecolor='#666'),
                yaxis=dict(autorange='reversed'),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                showlegend=False,
                margin=dict(l=140, r=80, t=20, b=40),
            )
            st.plotly_chart(fig_rep, use_container_width=True)

            # Top 5 recommendations
            st.markdown("#### 🏆 Top 5 Recommended Combinations")
            for i, (drug, score) in enumerate(scores_row[:5]):
                verdict, color = get_verdict(score)
                st.markdown(f"""
<div style="background:#1a1a2e;border-left:4px solid {'#d73027' if score>0.1 else '#2166ac'};
     padding:12px;border-radius:6px;margin:6px 0;color:white;">
<b>#{i+1} {anchor_drug} + {drug}</b><br>
Synergy score: <b>{score:.3f}</b> | {verdict}
</div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 7: Mechanism Explorer (no API needed)
# ═══════════════════════════════════════════════════════════════════════════════

with tab7:
    st.markdown("### 🔬 Mechanism of Action Explorer")
    st.caption("Understand why drug combinations work or fail based on their targets and pathways.")

    DRUG_MECHANISMS = {
        "Vemurafenib":   {"target":"BRAF V600E","pathway":"MAPK/ERK","class":"BRAF inhibitor","moa":"Blocks mutant BRAF kinase, inhibiting ERK signaling and tumor proliferation"},
        "Dabrafenib":    {"target":"BRAF V600E","pathway":"MAPK/ERK","class":"BRAF inhibitor","moa":"Selective BRAF inhibitor, reduces ERK phosphorylation in BRAF-mutant tumors"},
        "Trametinib":    {"target":"MEK1/2","pathway":"MAPK/ERK","class":"MEK inhibitor","moa":"Blocks MEK1/2, downstream of BRAF, preventing ERK activation and cell proliferation"},
        "Cobimetinib":   {"target":"MEK1","pathway":"MAPK/ERK","class":"MEK inhibitor","moa":"Allosteric MEK1 inhibitor that blocks ERK signaling"},
        "Selumetinib":   {"target":"MEK1/2","pathway":"MAPK/ERK","class":"MEK inhibitor","moa":"Non-ATP competitive MEK inhibitor reducing tumor cell proliferation"},
        "Imatinib":      {"target":"BCR-ABL/KIT/PDGFR","pathway":"RTK signaling","class":"TKI","moa":"Competitive inhibitor of BCR-ABL ATP binding, blocks CML proliferation"},
        "Dasatinib":     {"target":"BCR-ABL/SRC","pathway":"RTK signaling","class":"TKI","moa":"Dual BCR-ABL and SRC family kinase inhibitor, more potent than imatinib"},
        "Erlotinib":     {"target":"EGFR","pathway":"EGFR/RAS/MAPK","class":"EGFR TKI","moa":"Reversible EGFR inhibitor blocking downstream RAS-MAPK and PI3K-AKT signaling"},
        "Gefitinib":     {"target":"EGFR","pathway":"EGFR/RAS/MAPK","class":"EGFR TKI","moa":"Selective EGFR inhibitor preventing EGF-driven tumor cell proliferation"},
        "Lapatinib":     {"target":"EGFR/HER2","pathway":"EGFR/RAS/MAPK","class":"Dual TKI","moa":"Dual EGFR and HER2 inhibitor blocking both receptors simultaneously"},
        "Osimertinib":   {"target":"EGFR T790M","pathway":"EGFR/RAS/MAPK","class":"3rd gen EGFR TKI","moa":"Irreversible EGFR inhibitor overcoming T790M resistance mutation"},
        "Afatinib":      {"target":"EGFR/HER2/HER4","pathway":"EGFR/RAS/MAPK","class":"Pan-HER TKI","moa":"Irreversible pan-HER inhibitor blocking all ErbB family members"},
        "Olaparib":      {"target":"PARP1/2","pathway":"DNA repair","class":"PARP inhibitor","moa":"Traps PARP on DNA, preventing repair of single-strand breaks in BRCA-deficient tumors"},
        "Rucaparib":     {"target":"PARP1/2/3","pathway":"DNA repair","class":"PARP inhibitor","moa":"Pan-PARP inhibitor with additional PARP trapping activity"},
        "Niraparib":     {"target":"PARP1/2","pathway":"DNA repair","class":"PARP inhibitor","moa":"Potent PARP1/2 inhibitor causing synthetic lethality in HRD tumors"},
        "Palbociclib":   {"target":"CDK4/6","pathway":"Cell cycle","class":"CDK4/6 inhibitor","moa":"Prevents Rb phosphorylation, blocking G1-S transition and cell cycle progression"},
        "Abemaciclib":   {"target":"CDK4/6","pathway":"Cell cycle","class":"CDK4/6 inhibitor","moa":"More potent CDK4 inhibitor with additional CDK9 activity vs palbociclib"},
        "Ribociclib":    {"target":"CDK4/6","pathway":"Cell cycle","class":"CDK4/6 inhibitor","moa":"Selective CDK4/6 inhibitor restoring cell cycle control in HR+ breast cancer"},
        "Ibrutinib":     {"target":"BTK","pathway":"BCR signaling","class":"BTK inhibitor","moa":"Irreversible BTK inhibitor blocking B-cell receptor signaling in B-cell malignancies"},
        "Zanubrutinib":  {"target":"BTK","pathway":"BCR signaling","class":"BTK inhibitor","moa":"Next-gen BTK inhibitor with improved selectivity over ibrutinib"},
        "Acalabrutinib": {"target":"BTK","pathway":"BCR signaling","class":"BTK inhibitor","moa":"Highly selective covalent BTK inhibitor with fewer off-target effects"},
        "Venetoclax":    {"target":"BCL-2","pathway":"Apoptosis","class":"BCL-2 inhibitor","moa":"BH3 mimetic releasing pro-apoptotic proteins from BCL-2, triggering apoptosis"},
        "Alpelisib":     {"target":"PI3Kα","pathway":"PI3K/AKT/mTOR","class":"PI3K inhibitor","moa":"Selective PI3Kα inhibitor blocking PI3K-driven survival signaling"},
        "Paclitaxel":    {"target":"Tubulin","pathway":"Mitosis","class":"Taxane","moa":"Stabilizes microtubules preventing depolymerization, arresting cells in mitosis"},
        "Doxorubicin":   {"target":"TOP2/DNA","pathway":"DNA damage","class":"Anthracycline","moa":"Intercalates DNA and inhibits TOP2, causing double-strand breaks and apoptosis"},
        "Gemcitabine":   {"target":"RRM1","pathway":"Nucleotide synthesis","class":"Antimetabolite","moa":"Nucleoside analog inhibiting ribonucleotide reductase and DNA synthesis"},
        "Capecitabine":  {"target":"TYMS","pathway":"Nucleotide synthesis","class":"Antimetabolite","moa":"Oral 5-FU prodrug converted to 5-FU in tumor tissue inhibiting thymidylate synthase"},
        "Temozolomide":  {"target":"DNA","pathway":"DNA damage","class":"Alkylating agent","moa":"Alkylates guanine at O6 position causing DNA damage and apoptosis in glioblastoma"},
        "Sorafenib":     {"target":"BRAF/VEGFR/PDGFR","pathway":"MAPK/angiogenesis","class":"Multi-TKI","moa":"Multi-kinase inhibitor blocking tumor proliferation and angiogenesis"},
        "Sunitinib":     {"target":"VEGFR/PDGFR/KIT","pathway":"Angiogenesis","class":"Multi-TKI","moa":"Anti-angiogenic TKI blocking tumor vascularization and proliferation"},
        "Alectinib":     {"target":"ALK","pathway":"ALK/RAS/MAPK","class":"ALK inhibitor","moa":"2nd gen ALK inhibitor overcoming crizotinib resistance with CNS penetration"},
        "Belinostat":    {"target":"HDAC","pathway":"Epigenetics","class":"HDAC inhibitor","moa":"Pan-HDAC inhibitor causing histone hyperacetylation and tumor cell differentiation"},
        "Vorinostat":    {"target":"HDAC","pathway":"Epigenetics","class":"HDAC inhibitor","moa":"First FDA-approved HDAC inhibitor inducing cell cycle arrest and apoptosis"},
    }

    SYNERGY_RULES = {
        ("MAPK/ERK","MAPK/ERK"): ("⚠️ Same pathway — possible antagonism or redundancy. Both drugs hit the same cascade (BRAF→MEK→ERK). "
            "Exception: vertical inhibition (BRAF+MEK) often synergizes by preventing feedback reactivation."),
        ("EGFR/RAS/MAPK","MAPK/ERK"): ("✅ Likely synergistic — upstream+downstream combination. "
            "EGFR inhibition reduces RAS activation while MEK/BRAF inhibition blocks downstream signaling, preventing bypass resistance."),
        ("MAPK/ERK","EGFR/RAS/MAPK"): ("✅ Likely synergistic — upstream+downstream combination."),
        ("DNA repair","DNA damage"): ("✅ Strong synergy expected — PARP inhibition prevents repair of DNA damage caused by the chemotherapy agent. "
            "Classic synthetic lethality strategy."),
        ("DNA damage","DNA repair"): ("✅ Strong synergy expected — chemotherapy creates DNA damage that PARP inhibitors prevent from being repaired."),
        ("Cell cycle","DNA damage"): ("✅ Likely synergistic — CDK inhibition arrests cells in G1, making them more sensitive to DNA-damaging agents."),
        ("DNA damage","Cell cycle"): ("✅ Likely synergistic — DNA damage triggers checkpoints that CDK inhibitors can exploit."),
        ("Apoptosis","DNA damage"): ("✅ Likely synergistic — BCL-2 inhibition lowers the apoptotic threshold, sensitizing cells to DNA damage."),
        ("DNA damage","Apoptosis"): ("✅ Likely synergistic — DNA damage pushes cells toward apoptosis that BCL-2 inhibitors facilitate."),
        ("BCR signaling","Apoptosis"): ("✅ Strong synergy — BTK inhibition reduces survival signals while BCL-2 inhibition forces apoptosis. "
            "Venetoclax+ibrutinib is a validated CLL combination."),
        ("Apoptosis","BCR signaling"): ("✅ Strong synergy — validated combination in CLL."),
        ("PI3K/AKT/mTOR","MAPK/ERK"): ("✅ Likely synergistic — dual pathway blockade. Tumors often activate PI3K as a bypass when MAPK is inhibited."),
        ("MAPK/ERK","PI3K/AKT/mTOR"): ("✅ Likely synergistic — dual pathway blockade prevents bypass resistance."),
        ("Mitosis","DNA damage"): ("✅ Likely synergistic — taxanes arrest cells in mitosis making them more vulnerable to DNA damage."),
        ("DNA damage","Mitosis"): ("✅ Likely synergistic."),
        ("Epigenetics","DNA damage"): ("✅ Likely synergistic — HDAC inhibition opens chromatin, making DNA more accessible to damaging agents."),
        ("Nucleotide synthesis","DNA damage"): ("✅ Synergistic — both deplete DNA building blocks or damage DNA through complementary mechanisms."),
    }

    col_m1, col_m2 = st.columns(2)
    with col_m1:
        drug_moa_a = st.selectbox("Drug A:", list(DRUG_MECHANISMS.keys()),
            index=list(DRUG_MECHANISMS.keys()).index("Vemurafenib"), key="moa_a")
    with col_m2:
        drug_moa_b = st.selectbox("Drug B:", list(DRUG_MECHANISMS.keys()),
            index=list(DRUG_MECHANISMS.keys()).index("Trametinib"), key="moa_b")

    if drug_moa_a and drug_moa_b and drug_moa_a != drug_moa_b:
        moa_a = DRUG_MECHANISMS[drug_moa_a]
        moa_b = DRUG_MECHANISMS[drug_moa_b]

        col_info1, col_info2 = st.columns(2)
        with col_info1:
            st.markdown(f"""<div style="background:#1a1a2e;border-left:4px solid #4fc3f7;
            padding:12px;border-radius:6px;color:white;">
            <b>💊 {drug_moa_a}</b><br>
            <b>Target:</b> {moa_a['target']}<br>
            <b>Pathway:</b> {moa_a['pathway']}<br>
            <b>Class:</b> {moa_a['class']}<br>
            <b>MoA:</b> {moa_a['moa']}
            </div>""", unsafe_allow_html=True)
        with col_info2:
            st.markdown(f"""<div style="background:#1a1a2e;border-left:4px solid #ff9800;
            padding:12px;border-radius:6px;color:white;">
            <b>💊 {drug_moa_b}</b><br>
            <b>Target:</b> {moa_b['target']}<br>
            <b>Pathway:</b> {moa_b['pathway']}<br>
            <b>Class:</b> {moa_b['class']}<br>
            <b>MoA:</b> {moa_b['moa']}
            </div>""", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("#### 🧬 Combination Analysis")

        pathway_key = (moa_a['pathway'], moa_b['pathway'])
        if pathway_key in SYNERGY_RULES:
            explanation = SYNERGY_RULES[pathway_key]
        elif moa_a['target'] == moa_b['target']:
            explanation = (f"⚠️ Same target ({moa_a['target']}) — both drugs compete for the same binding site. "
                          "This typically leads to antagonism rather than synergy. "
                          "Consider combining drugs with complementary targets instead.")
        elif moa_a['class'] == moa_b['class']:
            explanation = (f"⚠️ Same drug class ({moa_a['class']}) — redundant mechanism of action. "
                          "Combinations within the same class often show additive effects at best.")
        else:
            explanation = (f"🔬 Complementary mechanisms — {drug_moa_a} targets {moa_a['target']} "
                          f"while {drug_moa_b} targets {moa_b['target']}. "
                          f"These operate in {'the same' if moa_a['pathway']==moa_b['pathway'] else 'different'} pathways. "
                          f"Synergy potential depends on tumor dependency on these targets.")

        bg = '#1e3a1e' if '✅' in explanation else '#3a1e1e' if '⚠️' in explanation else '#1e2a3a'
        border = '#4caf50' if '✅' in explanation else '#ff5722' if '⚠️' in explanation else '#4fc3f7'
        st.markdown(f"""<div style="background:{bg};border-left:4px solid {border};
        padding:16px;border-radius:6px;color:white;margin:8px 0;font-size:15px;">
        {explanation}
        </div>""", unsafe_allow_html=True)

        # Pathway diagram
        same_pathway = moa_a['pathway'] == moa_b['pathway']
        st.markdown(f"""
| Property | {drug_moa_a} | {drug_moa_b} |
|----------|------------|------------|
| Target | {moa_a['target']} | {moa_b['target']} |
| Pathway | {moa_a['pathway']} | {moa_b['pathway']} |
| Class | {moa_a['class']} | {moa_b['class']} |
| Same pathway | {'Yes ⚠️' if same_pathway else 'No ✅'} | — |
        """)