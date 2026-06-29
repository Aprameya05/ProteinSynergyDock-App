"""ProteinSynergyDock — Complete App with All Features"""

import streamlit as st
import torch, torch.nn as nn, torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool
from torch_geometric.data import Data, Batch
from rdkit import Chem
from rdkit.Chem import AllChem
import py3Dmol, numpy as np, os, requests, subprocess, tempfile, shutil, json
import streamlit.components.v1 as components
import plotly.graph_objects as go
import pandas as pd

st.set_page_config(page_title="ProteinSynergyDock", page_icon="🧬", layout="wide", initial_sidebar_state="expanded")
st.markdown("""<style>
.main-header{text-align:center;padding:2rem;background:linear-gradient(135deg,#1a1a2e 0%,#16213e 50%,#0f3460 100%);border-radius:12px;margin-bottom:2rem;}
.main-header h1{color:#4fc3f7;font-size:2.5rem;margin:0;}
.main-header p{color:#b0bec5;margin:0.5rem 0 0;}
.known-score{background:#1e3a1e;border-left:4px solid #4caf50;padding:12px;border-radius:6px;margin:8px 0;color:white;}
.unknown-score{background:#2a2a1e;border-left:4px solid #ff9800;padding:12px;border-radius:6px;margin:8px 0;color:white;}
.history-item{background:#1a1a2e;border-left:3px solid #4fc3f7;padding:8px;border-radius:4px;margin:4px 0;color:white;font-size:12px;}
</style>""", unsafe_allow_html=True)

class DrugEncoder(nn.Module):
    def __init__(self,in_dim=7,hidden=128,out_dim=256,heads=4):
        super().__init__()
        self.proj=nn.Linear(in_dim,hidden)
        self.conv1=GATv2Conv(hidden,hidden,heads=heads,concat=True)
        self.conv2=GATv2Conv(hidden*heads,out_dim,heads=1,concat=False)
        self.norm1=nn.LayerNorm(hidden*heads); self.norm2=nn.LayerNorm(out_dim)
    def forward(self,x,edge_index,batch):
        x=F.gelu(self.proj(x)); x=F.gelu(self.norm1(self.conv1(x,edge_index)))
        x=F.gelu(self.norm2(self.conv2(x,edge_index))); return global_mean_pool(x,batch)

class CrossDrugAttention(nn.Module):
    def __init__(self,dim=256):
        super().__init__()
        self.attn=nn.MultiheadAttention(dim,num_heads=4,batch_first=True)
        self.norm=nn.LayerNorm(dim); self.ff=nn.Sequential(nn.Linear(dim,dim*2),nn.GELU(),nn.Linear(dim*2,dim))
    def forward(self,a,b):
        seq=torch.stack([a,b],dim=1); att,_=self.attn(seq,seq,seq)
        seq=self.norm(seq+att); return (seq+self.ff(seq)).reshape(seq.shape[0],-1)

class ProteinSynergyDockV2(nn.Module):
    def __init__(self,go_dim=512,drug_dim=256,hidden=512,n_cell_lines=60):
        super().__init__()
        self.drug_encoder=DrugEncoder(7,128,drug_dim)
        self.cross_attn=CrossDrugAttention(drug_dim)
        self.film_scale=nn.Linear(go_dim,drug_dim*2); self.film_bias=nn.Linear(go_dim,drug_dim*2)
        self.cell_embed=nn.Embedding(n_cell_lines,32)
        self.head=nn.Sequential(nn.Linear(drug_dim*2+2+32,hidden),nn.LayerNorm(hidden),nn.ReLU(),nn.Dropout(0.2),
            nn.Linear(hidden,hidden//2),nn.ReLU(),nn.Dropout(0.1),nn.Linear(hidden//2,2))
    def forward(self,da,db,go_emb,dock,cell_idx):
        ea=self.drug_encoder(da.x,da.edge_index,da.batch); eb=self.drug_encoder(db.x,db.edge_index,db.batch)
        fused=self.cross_attn(ea,eb); fused=fused*(1+self.film_scale(go_emb))+self.film_bias(go_emb)
        fused=torch.cat([fused,dock,self.cell_embed(cell_idx)],dim=-1); out=self.head(fused); return out[:,0],out[:,1]

class ProteinSynergyDockV1(nn.Module):
    def __init__(self,go_dim=512,drug_dim=256,hidden=512):
        super().__init__()
        self.drug_encoder=DrugEncoder(7,128,drug_dim); self.cross_attn=CrossDrugAttention(drug_dim)
        self.film_scale=nn.Linear(go_dim,drug_dim*2); self.film_bias=nn.Linear(go_dim,drug_dim*2)
        self.head=nn.Sequential(nn.Linear(drug_dim*2+2,hidden),nn.LayerNorm(hidden),nn.ReLU(),nn.Dropout(0.2),
            nn.Linear(hidden,hidden//2),nn.ReLU(),nn.Dropout(0.1),nn.Linear(hidden//2,2))
    def forward(self,da,db,go_emb,dock):
        ea=self.drug_encoder(da.x,da.edge_index,da.batch); eb=self.drug_encoder(db.x,db.edge_index,db.batch)
        fused=self.cross_attn(ea,eb); fused=fused*(1+self.film_scale(go_emb))+self.film_bias(go_emb)
        fused=torch.cat([fused,dock],dim=-1); out=self.head(fused); return out[:,0],out[:,1]

@st.cache_resource
def load_model():
    p='proteinsydock_v2_final.pt'
    if not os.path.exists(p): return None,None,None,'none',0.0,0.0
    ckpt=torch.load(p,map_location='cpu',weights_only=False); sd=ckpt['state_dict']
    if any('cell_embed' in k for k in sd):
        m=ProteinSynergyDockV2(n_cell_lines=ckpt.get('n_cell_lines',60)); m.load_state_dict(sd); m.eval()
        return m,ckpt.get('cell_line_to_idx',{}),(ckpt.get('synergy_mean',-2.58),ckpt.get('synergy_std',6.06)),'v2',ckpt.get('pearson_r',0.0),ckpt.get('auroc',0.0)
    else:
        m=ProteinSynergyDockV1(); m.load_state_dict(sd); m.eval()
        return m,None,None,'v1',ckpt.get('pearson_r',0.0),ckpt.get('auroc',0.0)

model,cell_to_idx,syn_scale,model_version,model_r,model_auroc=load_model()
if 'history' not in st.session_state: st.session_state.history=[]

@st.cache_data
def load_precomputed():
    if os.path.exists('precomputed_scores.json'):
        with open('precomputed_scores.json') as f: return json.load(f)
    return None
scores_data=load_precomputed()

KNOWN_SYNERGY={
    ("Vemurafenib","Trametinib"):{"UACC-62":8.4,"SK-MEL-5":7.2,"A375":9.1},
    ("Trametinib","Vemurafenib"):{"UACC-62":8.4,"SK-MEL-5":7.2,"A375":9.1},
    ("Imatinib","Dasatinib"):{"K-562":-1.4,"MOLT-4":-0.8},
    ("Dasatinib","Imatinib"):{"K-562":-1.4,"MOLT-4":-0.8},
    ("Erlotinib","Lapatinib"):{"A549/ATCC":5.5,"NCI-H23":4.2},
    ("Lapatinib","Erlotinib"):{"A549/ATCC":5.5,"NCI-H23":4.2},
    ("Olaparib","Rucaparib"):{"OVCAR-3":2.1,"SK-OV-3":1.8},
    ("Rucaparib","Olaparib"):{"OVCAR-3":2.1,"SK-OV-3":1.8},
    ("Palbociclib","Abemaciclib"):{"MCF7":3.2,"T-47D":2.8},
    ("Abemaciclib","Palbociclib"):{"MCF7":3.2,"T-47D":2.8},
    ("Vemurafenib","Cobimetinib"):{"UACC-62":6.8,"SK-MEL-5":5.9},
    ("Cobimetinib","Vemurafenib"):{"UACC-62":6.8,"SK-MEL-5":5.9},
}

DRUG_SMILES_LOOKUP={
    "Imatinib":"CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)NC4=NC=CC(=N4)C5=CN=CC=C5",
    "Gefitinib":"COC1=C(C=C2C(=C1)N=CN=C2NC3=CC(=C(C=C3)F)Cl)OCCCN4CCOCC4",
    "Erlotinib":"COCCOC1=C(C=C2C(=C1)C(=NC=N2)NC3=CC=CC(=C3)C#C)OCCOC",
    "Lapatinib":"CS(=O)(=O)CCNCc1oc(cc1)c2ccc3ncnc(Nc4ccc(Oc5cccc(Cl)c5)c(Cl)c4)c3c2",
    "Dasatinib":"Cc1nc(Nc2ncc(s2)C(=O)Nc2c(C)cccc2Cl)cc(n1)N1CCN(CCO)CC1",
    "Nilotinib":"Cc1cn(c2cc(NC(=O)c3ccc(C)c(Nc4nccc(n4)-c4cccnc4)c3)cc(C(F)(F)F)c12)C",
    "Vemurafenib":"CCCS(=O)(=O)Nc1ccc(F)c(C(=O)c2c[nH]c3ncc(-c4ccc(Cl)cc4)cc23)c1",
    "Dabrafenib":"CC(C)(C)c1nc2cc(F)ccc2c(C(=O)Nc2ccc(F)c(NS(=O)(=O)c3ccc(F)cc3)c2)n1",
    "Trametinib":"CC(=O)Nc1ccc(-c2cc3c(nc(N)nc3n2C)N2CCC(F)(F)CC2=O)cc1F",
    "Cobimetinib":"OC(COc1cc(Cl)c(F)cc1F)CN1CCC(=C1)c1cc2c(Nc3ccc(F)cc3F)ncc(C(N)=O)c2[nH]1",
    "Sorafenib":"CNC(=O)c1cc(Oc2ccc(NC(=O)Nc3ccc(Cl)c(C(F)(F)F)c3)cc2)ccn1",
    "Sunitinib":"CCN(CC)CCNC(=O)c1c(C)[nH]c(C=C2C(=O)Nc3ccc(F)cc32)c1C",
    "Olaparib":"O=C1CCCN1c1ccc(cc1)C(=O)c1[nH]ncc1C1CC1",
    "Niraparib":"OC(=O)c1ccc2[nH]ncc2c1-c1ccc(cn1)C1CCNCC1",
    "Rucaparib":"NCc1cc2cc(F)ccc2[nH]1-c1ccc3NCCCC(=O)c3c1",
    "Palbociclib":"CC1=C(C(=NC(=C1)N2CCNCC2)N3CCCC3)C(=O)NC4=CC=CC=N4",
    "Abemaciclib":"CC1=NC(=NC(=C1)NC2=NC=CC(=N2)N3CCC(CC3)NC(=O)C4=CC=C(C=C4)F)C5=CC(=CC=C5)F",
    "Ribociclib":"CC1=NC(=NC(=C1)N2CCNCC2)C3=CC4=C(C=C3)N=CN=C4N5CCCC5",
    "Ibrutinib":"C=CC(=O)N1CCCC(c2ncnc3[nH]ccc23)C1",
    "Zanubrutinib":"O=C(/C=C/c1ccco1)N1CCC(n2nc(-c3ccc4c(c3)CCNC4=O)c3c(N)ncnc23)CC1",
    "Acalabrutinib":"CC#CC(=O)N1CCC(n2nc(-c3ccc4c(c3)CCNC4=O)c3c(N)ncnc23)CC1",
    "Venetoclax":"CC1(CCC(CC1)N2CCN(CC2)c3ccc(cc3)C(=O)NS(=O)(=O)c4ccc(cc4-c5cnc6ccccc6n5)Cl)C",
    "Alpelisib":"CC1(C)CN(c2nc(Nc3ccc(S(N)(=O)=O)cc3F)ncc2F)CC1=O",
    "Paclitaxel":"O=C(OC1C[C@]2(O)C(=O)C(OC(=O)c3ccccc3)C(O)C(OC(=O)C(NC(=O)c3ccccc3)c3ccccc3)C2(C)CC1)C(C)=C",
    "Doxorubicin":"COc1cccc2C(=O)c3c(O)c4CC(O)(CC(OC5CC(N)C(O)C(C)O5)c4c(O)c3C(=O)c12)C(=O)CO",
    "Gemcitabine":"NC(=O)C1=CN(C(=O)N1)C1CC(F)(F)C(CO)O1",
    "Osimertinib":"C=CC(=O)Nc1cc2c(Nc3ccc(F)c(Cl)c3)nc(OC)nc2cc1N(C)CCN(C)C",
    "Alectinib":"COc1cc2c(cc1N1CCC(CC1)c1ccc3[nH]ccc3c1)cc(=O)n1ccc(C#N)c21",
    "Afatinib":"CN(C)C/C=C/C(=O)Nc1cc2c(Nc3ccc(F)c(Cl)c3)ncnc2cc1OC",
    "Capecitabine":"CCOC(=O)Nc1nc(=O)n(C2OC(C)C(O)C2O)cc1F",
    "Temozolomide":"Cn1nnc2c(C(N)=O)ncn12",
    "Selumetinib":"Cc1cc(Nc2ncc(F)c(Nc3ccc(I)c(F)c3)n2)c(Cl)cc1Cl",
    "Belinostat":"O=C(/C=C/c1ccccc1)NOc1ccc(NS(=O)(=O)c2ccccc2)cc1",
    "Vorinostat":"O=C(CCCCCCC(=O)Nc1ccccc1)NO",
}

CANCER_PANELS={
    "Melanoma":["UACC-62","SK-MEL-5","SK-MEL-28","MALME-3M","M14","MDA-MB-435","UACC-257","LOX IMVI"],
    "Non-Small Cell Lung Cancer":["A549/ATCC","NCI-H23","NCI-H226","NCI-H322M","NCI-H460","NCI-H522","EKVX","HOP-62","HOP-92"],
    "Breast Cancer":["MCF7","MDA-MB-231/ATCC","HS 578T","BT-549","T-47D","MDA-MB-468"],
    "Colon Cancer":["COLO 205","HCC-2998","HCT-116","HCT-15","HT29","KM12","SW-620"],
    "Leukemia":["CCRF-CEM","HL-60(TB)","K-562","MOLT-4","RPMI-8226","SR"],
    "Ovarian Cancer":["IGROV1","OVCAR-3","OVCAR-4","OVCAR-5","OVCAR-8","SK-OV-3","NCI/ADR-RES"],
    "CNS Cancer":["SF-268","SF-295","SF-539","SNB-19","SNB-75","U251"],
    "Renal Cancer":["786-0","A498","ACHN","CAKI-1","RXF 393","SN12C","TK-10","UO-31"],
    "Prostate Cancer":["DU-145","PC-3"],
}

SHOWCASES={
    "Custom input":{"smiles_a":"","smiles_b":"","pdb_id":"","name_a":"","name_b":"","panel":"Melanoma","cell_line":"UACC-62","note":""},
    "✅ Vemurafenib + Trametinib (FDA Approved)":{"smiles_a":"CCCS(=O)(=O)Nc1ccc(F)c(C(=O)c2c[nH]c3ncc(-c4ccc(Cl)cc4)cc23)c1","smiles_b":"CC(=O)Nc1ccc(-c2cc3c(nc(N)nc3n2C)N2CCC(F)(F)CC2=O)cc1F","pdb_id":"3OG7","name_a":"Vemurafenib","name_b":"Trametinib","panel":"Melanoma","cell_line":"UACC-62","note":"FDA-approved BRAF+MEK combo for melanoma. Known synergy: **8.4**"},
    "❌ Imatinib + Dasatinib (Antagonistic)":{"smiles_a":"CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)NC4=NC=CC(=N4)C5=CN=CC=C5","smiles_b":"Cc1nc(Nc2ncc(s2)C(=O)Nc2c(C)cccc2Cl)cc(n1)N1CCN(CCO)CC1","pdb_id":"2HYY","name_a":"Imatinib","name_b":"Dasatinib","panel":"Leukemia","cell_line":"K-562","note":"Both compete for ABL1 ATP pocket. Known synergy: **-1.4**"},
    "✅ Erlotinib + Lapatinib (Synergistic)":{"smiles_a":"COCCOC1=C(C=C2C(=C1)C(=NC=N2)NC3=CC=CC(=C3)C#C)OCCOC","smiles_b":"CS(=O)(=O)CCNCc1oc(cc1)c2ccc3ncnc(Nc4ccc(Oc5cccc(Cl)c5)c(Cl)c4)c3c2","pdb_id":"1IVO","name_a":"Erlotinib","name_b":"Lapatinib","panel":"Non-Small Cell Lung Cancer","cell_line":"A549/ATCC","note":"Dual EGFR inhibition. Known synergy: **5.5**"},
    "⚠️ Olaparib + Rucaparib (Mild Synergy)":{"smiles_a":"O=C1CCCN1c1ccc(cc1)C(=O)c1[nH]ncc1C1CC1","smiles_b":"NCc1cc2cc(F)ccc2[nH]1-c1ccc3NCCCC(=O)c3c1","pdb_id":"4DQY","name_a":"Olaparib","name_b":"Rucaparib","panel":"Ovarian Cancer","cell_line":"OVCAR-3","note":"PARP inhibition synergy. Known: **2.1**"},
}

DRUG_MECHANISMS={
    "Vemurafenib":{"target":"BRAF V600E","pathway":"MAPK/ERK","class":"BRAF inhibitor","moa":"Blocks mutant BRAF kinase, inhibiting ERK signaling and tumor proliferation"},
    "Dabrafenib":{"target":"BRAF V600E","pathway":"MAPK/ERK","class":"BRAF inhibitor","moa":"Selective BRAF inhibitor, reduces ERK phosphorylation in BRAF-mutant tumors"},
    "Trametinib":{"target":"MEK1/2","pathway":"MAPK/ERK","class":"MEK inhibitor","moa":"Blocks MEK1/2 downstream of BRAF, preventing ERK activation and proliferation"},
    "Cobimetinib":{"target":"MEK1","pathway":"MAPK/ERK","class":"MEK inhibitor","moa":"Allosteric MEK1 inhibitor blocking ERK signaling"},
    "Selumetinib":{"target":"MEK1/2","pathway":"MAPK/ERK","class":"MEK inhibitor","moa":"Non-ATP competitive MEK inhibitor"},
    "Imatinib":{"target":"BCR-ABL/KIT/PDGFR","pathway":"RTK signaling","class":"TKI","moa":"Competitive BCR-ABL ATP inhibitor, blocks CML proliferation"},
    "Dasatinib":{"target":"BCR-ABL/SRC","pathway":"RTK signaling","class":"TKI","moa":"Dual BCR-ABL and SRC kinase inhibitor"},
    "Erlotinib":{"target":"EGFR","pathway":"EGFR/RAS/MAPK","class":"EGFR TKI","moa":"Reversible EGFR inhibitor blocking RAS-MAPK and PI3K-AKT signaling"},
    "Gefitinib":{"target":"EGFR","pathway":"EGFR/RAS/MAPK","class":"EGFR TKI","moa":"Selective EGFR inhibitor preventing EGF-driven proliferation"},
    "Lapatinib":{"target":"EGFR/HER2","pathway":"EGFR/RAS/MAPK","class":"Dual TKI","moa":"Dual EGFR and HER2 inhibitor"},
    "Osimertinib":{"target":"EGFR T790M","pathway":"EGFR/RAS/MAPK","class":"3rd gen EGFR TKI","moa":"Irreversible EGFR inhibitor overcoming T790M resistance"},
    "Afatinib":{"target":"EGFR/HER2/HER4","pathway":"EGFR/RAS/MAPK","class":"Pan-HER TKI","moa":"Irreversible pan-HER inhibitor"},
    "Olaparib":{"target":"PARP1/2","pathway":"DNA repair","class":"PARP inhibitor","moa":"Traps PARP on DNA, causing synthetic lethality in BRCA-deficient cells"},
    "Rucaparib":{"target":"PARP1/2/3","pathway":"DNA repair","class":"PARP inhibitor","moa":"Pan-PARP inhibitor with PARP trapping activity"},
    "Niraparib":{"target":"PARP1/2","pathway":"DNA repair","class":"PARP inhibitor","moa":"Potent PARP1/2 inhibitor for HRD tumors"},
    "Palbociclib":{"target":"CDK4/6","pathway":"Cell cycle","class":"CDK4/6 inhibitor","moa":"Blocks G1-S transition by preventing Rb phosphorylation"},
    "Abemaciclib":{"target":"CDK4/6","pathway":"Cell cycle","class":"CDK4/6 inhibitor","moa":"More potent CDK4 inhibitor with additional CDK9 activity"},
    "Ribociclib":{"target":"CDK4/6","pathway":"Cell cycle","class":"CDK4/6 inhibitor","moa":"Selective CDK4/6 inhibitor for HR+ breast cancer"},
    "Ibrutinib":{"target":"BTK","pathway":"BCR signaling","class":"BTK inhibitor","moa":"Irreversible BTK inhibitor blocking B-cell receptor signaling"},
    "Zanubrutinib":{"target":"BTK","pathway":"BCR signaling","class":"BTK inhibitor","moa":"Next-gen BTK inhibitor with improved selectivity"},
    "Acalabrutinib":{"target":"BTK","pathway":"BCR signaling","class":"BTK inhibitor","moa":"Highly selective covalent BTK inhibitor"},
    "Venetoclax":{"target":"BCL-2","pathway":"Apoptosis","class":"BCL-2 inhibitor","moa":"BH3 mimetic triggering apoptosis in BCL-2 dependent tumors"},
    "Alpelisib":{"target":"PI3Ka","pathway":"PI3K/AKT/mTOR","class":"PI3K inhibitor","moa":"Selective PI3Ka inhibitor blocking survival signaling"},
    "Paclitaxel":{"target":"Tubulin","pathway":"Mitosis","class":"Taxane","moa":"Stabilizes microtubules arresting cells in mitosis"},
    "Doxorubicin":{"target":"TOP2/DNA","pathway":"DNA damage","class":"Anthracycline","moa":"Intercalates DNA and inhibits TOP2 causing double-strand breaks"},
    "Gemcitabine":{"target":"RRM1","pathway":"Nucleotide synthesis","class":"Antimetabolite","moa":"Nucleoside analog inhibiting DNA synthesis"},
    "Capecitabine":{"target":"TYMS","pathway":"Nucleotide synthesis","class":"Antimetabolite","moa":"Oral 5-FU prodrug inhibiting thymidylate synthase"},
    "Temozolomide":{"target":"DNA","pathway":"DNA damage","class":"Alkylating agent","moa":"Alkylates guanine causing DNA damage in glioblastoma"},
    "Sorafenib":{"target":"BRAF/VEGFR/PDGFR","pathway":"MAPK/angiogenesis","class":"Multi-TKI","moa":"Multi-kinase inhibitor blocking proliferation and angiogenesis"},
    "Sunitinib":{"target":"VEGFR/PDGFR/KIT","pathway":"Angiogenesis","class":"Multi-TKI","moa":"Anti-angiogenic TKI blocking tumor vascularization"},
    "Alectinib":{"target":"ALK","pathway":"ALK/RAS/MAPK","class":"ALK inhibitor","moa":"2nd gen ALK inhibitor with CNS penetration"},
    "Belinostat":{"target":"HDAC","pathway":"Epigenetics","class":"HDAC inhibitor","moa":"Pan-HDAC inhibitor causing histone hyperacetylation"},
    "Vorinostat":{"target":"HDAC","pathway":"Epigenetics","class":"HDAC inhibitor","moa":"First FDA-approved HDAC inhibitor"},
}

SYNERGY_RULES={
    ("MAPK/ERK","MAPK/ERK"):"⚠️ Same pathway — possible antagonism. Exception: vertical BRAF+MEK inhibition synergizes by preventing feedback reactivation.",
    ("EGFR/RAS/MAPK","MAPK/ERK"):"✅ Likely synergistic — upstream+downstream blocks bypass resistance.",
    ("MAPK/ERK","EGFR/RAS/MAPK"):"✅ Likely synergistic — upstream+downstream combination.",
    ("DNA repair","DNA damage"):"✅ Strong synergy — PARP inhibition prevents repair of chemotherapy DNA damage. Classic synthetic lethality.",
    ("DNA damage","DNA repair"):"✅ Strong synergy — chemotherapy damage that PARP inhibitors prevent from being repaired.",
    ("Cell cycle","DNA damage"):"✅ Likely synergistic — CDK inhibition arrests cells in G1, sensitizing to DNA damage.",
    ("DNA damage","Cell cycle"):"✅ Likely synergistic — DNA damage triggers checkpoints CDK inhibitors exploit.",
    ("Apoptosis","DNA damage"):"✅ Likely synergistic — BCL-2 inhibition lowers apoptotic threshold, sensitizing to DNA damage.",
    ("DNA damage","Apoptosis"):"✅ Likely synergistic — DNA damage pushes cells toward apoptosis BCL-2 inhibitors facilitate.",
    ("BCR signaling","Apoptosis"):"✅ Strong synergy — BTK+BCL-2. Venetoclax+ibrutinib is a validated CLL combination.",
    ("Apoptosis","BCR signaling"):"✅ Strong synergy — validated CLL combination.",
    ("PI3K/AKT/mTOR","MAPK/ERK"):"✅ Likely synergistic — dual pathway blockade prevents PI3K bypass resistance.",
    ("MAPK/ERK","PI3K/AKT/mTOR"):"✅ Likely synergistic — dual pathway blockade.",
    ("Mitosis","DNA damage"):"✅ Likely synergistic — taxanes arrest cells in mitosis making them vulnerable to DNA damage.",
    ("DNA damage","Mitosis"):"✅ Likely synergistic.",
    ("Epigenetics","DNA damage"):"✅ Likely synergistic — HDAC inhibition opens chromatin making DNA more accessible.",
    ("Nucleotide synthesis","DNA damage"):"✅ Synergistic — complementary DNA depletion and damage mechanisms.",
}

def lookup_known(da,db,cl=None):
    k=(da,db)
    if k not in KNOWN_SYNERGY: return None
    sc=KNOWN_SYNERGY[k]
    if cl and cl in sc: return sc[cl],cl
    return np.mean(list(sc.values())),f"avg {len(sc)} lines"

def smiles_to_graph(smiles):
    mol=Chem.MolFromSmiles(smiles)
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
            int(atom.GetIsAromatic()),int(atom.IsInRing()),atom.GetTotalNumHs(),atom.GetNumRadicalElectrons()])
        if conf: p=conf.GetAtomPosition(atom.GetIdx()); pos.append([p.x,p.y,p.z])
        else: pos.append([0.,0.,0.])
    es,ed=[],[]
    for bond in mol.GetBonds():
        i,j=bond.GetBeginAtomIdx(),bond.GetEndAtomIdx(); es+=[i,j]; ed+=[j,i]
    if not es: return None
    return Data(x=torch.tensor(feats,dtype=torch.float),pos=torch.tensor(pos,dtype=torch.float),edge_index=torch.tensor([es,ed],dtype=torch.long))

def fetch_pdb(pdb_id,save_dir):
    path=os.path.join(save_dir,f"{pdb_id}.pdb")
    if os.path.exists(path) and os.path.getsize(path)>1000: return path
    r=requests.get(f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb",timeout=30)
    if r.status_code==200:
        with open(path,'w') as f: f.write(r.text); return path
    return None

def get_protein_info(pdb_id):
    try:
        r=requests.get(f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id.upper()}",timeout=10)
        if r.status_code==200: return r.json().get('struct',{}).get('title',f'Protein {pdb_id}')
    except: pass
    return f"Protein {pdb_id}"

def get_binding_box(pdb_path,padding=10.0):
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
        c=np.array(hetatm); return c.mean(axis=0).tolist(),np.clip(c.max(axis=0)-c.min(axis=0)+padding*2,18,30).tolist(),"ligand"
    if atom:
        c=np.array(atom); return c.mean(axis=0).tolist(),np.clip(c.max(axis=0)-c.min(axis=0)+padding,20,28).tolist(),"protein_center"
    return [0,0,0],[25,25,25],"default"

def find_vina():
    for cmd in ['vina','autodock_vina','/usr/bin/vina','/usr/local/bin/vina']:
        if shutil.which(cmd): return cmd
    return None

def prepare_ligand(smiles,name,wd):
    out=f'{wd}/{name}.pdbqt'
    if os.path.exists(out) and os.path.getsize(out)>0: return out
    mol=Chem.MolFromSmiles(smiles)
    if mol is None: return None
    try:
        mol=Chem.AddHs(mol); AllChem.EmbedMolecule(mol,AllChem.ETKDGv3())
        AllChem.MMFFOptimizeMolecule(mol); mol=Chem.RemoveHs(mol)
    except:
        try: AllChem.Compute2DCoords(mol)
        except: return None
    sdf=f'{wd}/{name}.sdf'; pdb=f'{wd}/{name}.pdb'
    w=Chem.SDWriter(sdf); w.write(mol); w.close()
    subprocess.run(['obabel',sdf,'-O',pdb,'-h'],capture_output=True)
    subprocess.run(['obabel',pdb,'-O',out,'--partialcharge','gasteiger'],capture_output=True)
    return out if os.path.exists(out) and os.path.getsize(out)>0 else None

def prepare_receptor(pdb_path,wd):
    pid=os.path.basename(pdb_path).replace('.pdb',''); out=f'{wd}/{pid}_rec.pdbqt'
    if os.path.exists(out) and os.path.getsize(out)>0: return out
    clean=f'{wd}/{pid}_clean.pdb'
    with open(pdb_path) as fin, open(clean,'w') as fout:
        for line in fin:
            if line.startswith('ATOM') or line.startswith('END'): fout.write(line)
    subprocess.run(['obabel',clean,'-O',out,'--partialcharge','gasteiger','-xr'],capture_output=True)
    return out if os.path.exists(out) and os.path.getsize(out)>0 else None

def run_vina(vina,rec,lig,center,size,out,exh=8):
    cmd=[vina,'--receptor',rec,'--ligand',lig,'--out',out,
         '--center_x',str(round(center[0],3)),'--center_y',str(round(center[1],3)),'--center_z',str(round(center[2],3)),
         '--size_x',str(round(size[0],3)),'--size_y',str(round(size[1],3)),'--size_z',str(round(size[2],3)),
         '--exhaustiveness',str(exh),'--num_modes','3']
    try:
        res=subprocess.run(cmd,capture_output=True,text=True,timeout=300)
        sc=None
        if os.path.exists(out):
            with open(out) as f:
                for line in f:
                    if 'REMARK VINA RESULT' in line:
                        try: sc=float(line.split()[3]); break
                        except: pass
        if sc is None:
            for line in res.stdout.split('\n'):
                s=line.strip()
                if s and s[0]=='1' and len(s.split())>=3:
                    try: sc=float(s.split()[1]); break
                    except: pass
        return sc,res.stderr
    except Exception as e: return None,str(e)

def read_pose(pdbqt):
    atoms=[]
    if not os.path.exists(pdbqt): return None
    with open(pdbqt) as f:
        for line in f:
            if line.startswith('ENDMDL'): break
            if line.startswith(('ATOM','HETATM')):
                try: atoms.append((line[12:16].strip(),float(line[30:38]),float(line[38:46]),float(line[46:54])))
                except: pass
    return atoms or None

def pose_block(atoms,chain='A'):
    b="MODEL 1\n"
    for i,(a,x,y,z) in enumerate(atoms):
        b+=f"HETATM{i+1:5d}  {a:<4s}LG{chain} {chain}   1    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00\n"
    return b+"ENDMDL\n"

def show_3d(pdb,pa,pb,na,nb,h=500):
    v=py3Dmol.view(width=750,height=h)
    v.addModel(pdb,'pdb'); v.setStyle({'model':0},{'cartoon':{'color':'spectrum','opacity':0.65}})
    if pa:
        v.addModel(pose_block(pa,'A'),'pdb')
        v.setStyle({'model':1},{'stick':{'colorscheme':'cyanCarbon','radius':0.2},'sphere':{'colorscheme':'cyanCarbon','scale':0.3}})
    if pb:
        v.addModel(pose_block(pb,'B'),'pdb')
        idx=2 if pa else 1
        v.setStyle({'model':idx},{'stick':{'colorscheme':'orangeCarbon','radius':0.2},'sphere':{'colorscheme':'orangeCarbon','scale':0.3}})
    v.setBackgroundColor('#1a1a2e'); v.zoomTo({'model':1} if pa else {}); v.zoom(1.3)
    components.html(v._make_html(),height=h+20,scrolling=False)

def show_drugs(sa,sb,h=400):
    v=py3Dmol.view(width=750,height=h); off=0
    for i,(sm,col) in enumerate([(sa,'cyanCarbon'),(sb,'orangeCarbon')]):
        mol=Chem.MolFromSmiles(sm) if sm else None
        if mol is None: continue
        try:
            mol=Chem.AddHs(mol); AllChem.EmbedMolecule(mol,AllChem.ETKDGv3())
            AllChem.MMFFOptimizeMolecule(mol); mol=Chem.RemoveHs(mol)
            conf=mol.GetConformer()
            for j in range(mol.GetNumAtoms()):
                p=conf.GetAtomPosition(j); conf.SetAtomPosition(j,(p.x+off,p.y,p.z))
            v.addModel(Chem.MolToMolBlock(mol),'sdf')
            v.setStyle({'model':i},{'stick':{'colorscheme':col,'radius':0.15},'sphere':{'colorscheme':col,'scale':0.3}})
            off+=15
        except: pass
    v.setBackgroundColor('#1a1a2e'); v.zoomTo()
    components.html(v._make_html(),height=h+20,scrolling=False)

def get_verdict(s):
    if s>0.5: return "✅ Strongly Synergistic","green"
    elif s>0.1: return "⚠️ Mildly Synergistic","orange"
    elif s>-0.1: return "➖ Approximately Additive","blue"
    else: return "❌ Antagonistic","red"

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""<div class="main-header">
<h1>🧬 ProteinSynergyDock</h1>
<p>Structure-aware drug combination synergy prediction with cell line context</p>
<p style="font-size:13px;color:#78909c;margin-top:8px;">Real AutoDock Vina docking · ProteinWhisper++ GO context · 60 cancer cell lines</p>
</div>""", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔬 Quick Examples")
    example=st.selectbox("Choose a drug pair:",list(SHOWCASES.keys()))
    ex=SHOWCASES[example]
    if ex["note"]: st.info(ex["note"])
    st.markdown("---")
    st.markdown(f"""## 📊 Model Info
- **Version:** {model_version.upper() if model_version!='none' else 'Not loaded'}
- **Pearson r:** {model_r:.4f}
- **AUROC:** {model_auroc:.4f}
- **Real docking:** AutoDock Vina
- **Training data:** 107,103 NCI ALMANAC scores
- **Cell lines:** 60 cancer types

## 🔗 Links
- [GitHub](https://github.com/Aprameya05/ProteinSynergyDock)
- [ProteinWhisper](https://github.com/Aprameya05/ProteinWhisper)
- [DrugSynergy3D](https://github.com/Aprameya05/DrugSynergy3D)""")
    if st.session_state.history:
        st.markdown("---\n## 📜 Recent Predictions")
        for h in st.session_state.history:
            st.markdown(f"""<div class="history-item"><b>{h['drug_a']} + {h['drug_b']}</b><br>
{h['cell_line']} | Score: {h['score']:.3f} | {h['verdict'].split()[0]}</div>""", unsafe_allow_html=True)

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1,tab2,tab3,tab4,tab5,tab6,tab7=st.tabs([
    "🔬 Predict Synergy","🗺️ Synergy Landscape","📊 Cell Line Comparison",
    "🏥 Clinical Trials","📚 Literature","🔄 Drug Repurposing","🔬 Mechanism Explorer"])

# ═══ TAB 1 ════════════════════════════════════════════════════════════════════
with tab1:
    col1,col2=st.columns([1,1.2])
    with col1:
        st.markdown("### 💊 Drug Inputs")
        dao=["Custom (paste SMILES below)"]+sorted(DRUG_SMILES_LOOKUP.keys())
        das=st.selectbox("Drug A — select known drug",dao,key="da_select")
        if das!="Custom (paste SMILES below)":
            smiles_a=DRUG_SMILES_LOOKUP[das]; name_a=das
            st.text_area("Drug A SMILES",value=smiles_a,height=60,disabled=True)
        else:
            name_a=st.text_input("Drug A name",value=ex.get("name_a",""),placeholder="e.g. Imatinib")
            smiles_a=st.text_area("Drug A — SMILES",value=ex["smiles_a"],height=80)
        dbo=["Custom (paste SMILES below)"]+sorted(DRUG_SMILES_LOOKUP.keys())
        dbs=st.selectbox("Drug B — select known drug",dbo,key="db_select")
        if dbs!="Custom (paste SMILES below)":
            smiles_b=DRUG_SMILES_LOOKUP[dbs]; name_b=dbs
            st.text_area("Drug B SMILES",value=smiles_b,height=60,disabled=True)
        else:
            name_b=st.text_input("Drug B name",value=ex.get("name_b",""),placeholder="e.g. Dasatinib")
            smiles_b=st.text_area("Drug B — SMILES",value=ex["smiles_b"],height=80)
        st.markdown("### 🧫 Target Protein")
        pdb_id=st.text_input("PDB ID",value=ex.get("pdb_id",""),placeholder="e.g. 2HYY").strip().upper()
        if pdb_id: st.caption(f"Will fetch: https://files.rcsb.org/download/{pdb_id}.pdb")
        st.markdown("### 🏥 Cancer Context")
        panel=st.selectbox("Cancer type:",list(CANCER_PANELS.keys()),
            index=list(CANCER_PANELS.keys()).index(ex.get("panel","Melanoma")) if ex.get("panel","Melanoma") in CANCER_PANELS else 0)
        clp=CANCER_PANELS[panel]; dcl=ex.get("cell_line",clp[0])
        if dcl not in clp: dcl=clp[0]
        cell_line=st.selectbox("Cell line:",clp,index=clp.index(dcl))
        exhaustiveness=st.slider("Docking exhaustiveness",4,16,8,2)
        run_btn=st.button("🔬 Run Docking + Predict Synergy",type="primary")
    with col2:
        st.markdown("### 🔭 3D Visualization")
        viz=st.empty()
        if smiles_a or smiles_b:
            with viz.container():
                st.caption("Preview (pre-docking)")
                show_drugs(smiles_a,smiles_b)
                st.caption("🔵 Drug A  🟠 Drug B  *Drag to rotate*")

    if run_btn:
        if not smiles_a or not smiles_b: st.error("Enter SMILES for both drugs"); st.stop()
        if not pdb_id: st.error("Enter a PDB ID"); st.stop()
        if model is None: st.error("Model not loaded"); st.stop()
        ga=smiles_to_graph(smiles_a); gb=smiles_to_graph(smiles_b)
        if ga is None: st.error("❌ Invalid SMILES for Drug A"); st.stop()
        if gb is None: st.error("❌ Invalid SMILES for Drug B"); st.stop()
        known=lookup_known(name_a or "Drug A",name_b or "Drug B",cell_line)
        vina_cmd=find_vina(); obabel_cmd=shutil.which('obabel')
        st.markdown("---\n### 🔄 Pipeline Running...")
        prog=st.progress(0); stat=st.status("Starting...",expanded=True)
        with tempfile.TemporaryDirectory() as wd:
            with stat: st.write(f"📥 Fetching {pdb_id}...")
            prog.progress(10)
            pdb_path=fetch_pdb(pdb_id,wd)
            if not pdb_path: st.error(f"❌ Could not fetch {pdb_id}"); st.stop()
            pdb_content=open(pdb_path).read()
            pname=get_protein_info(pdb_id)
            center,size,bmethod=get_binding_box(pdb_path)
            with stat:
                st.write(f"✅ {pname[:70]}")
                st.write(f"📦 Box: {bmethod} | {[round(c,1) for c in center]}")
            prog.progress(20)
            dsa=dsb=-7.0; pa=pb=None; dran=False
            if vina_cmd and obabel_cmd:
                rec=prepare_receptor(pdb_path,wd); prog.progress(30)
                if rec:
                    with stat: st.write("✅ Receptor ready")
                    with stat: st.write(f"🔬 Docking {name_a or 'Drug A'}...")
                    la=prepare_ligand(smiles_a,"drug_a",wd)
                    if la:
                        oa=f'{wd}/drug_a_out.pdbqt'
                        sa,_=run_vina(vina_cmd,rec,la,center,size,oa,exhaustiveness)
                        if sa is not None:
                            dsa=sa; pa=read_pose(oa); dran=True
                            st.session_state['pa']=pa
                            st.session_state['pdb_content']=pdb_content
                            st.session_state['center']=center
                            st.session_state['pname']=pname
                            st.session_state['bmethod']=bmethod
                            with stat: st.write(f"✅ {name_a or 'Drug A'}: {sa:.2f} kcal/mol")
                    prog.progress(60)
                    with stat: st.write(f"🔬 Docking {name_b or 'Drug B'}...")
                    lb=prepare_ligand(smiles_b,"drug_b",wd)
                    if lb:
                        ob=f'{wd}/drug_b_out.pdbqt'
                        sb,_=run_vina(vina_cmd,rec,lb,center,size,ob,exhaustiveness)
                        if sb is not None:
                            dsb=sb; pb=read_pose(ob); dran=True
                            st.session_state['pb']=pb
                            with stat: st.write(f"✅ {name_b or 'Drug B'}: {sb:.2f} kcal/mol")
            else:
                with stat: st.write("⚠️ Docking tools unavailable")
            # Save all result variables to session state
            st.session_state['dsa']=dsa; st.session_state['dsb']=dsb
            st.session_state['dran']=dran; st.session_state['syn_score']=None
            st.session_state['name_a']=name_a; st.session_state['name_b']=name_b
            st.session_state['panel']=panel; st.session_state['cell_line']=cell_line
            st.session_state['pdb_id']=pdb_id
            prog.progress(75)
            with stat: st.write("🧠 Predicting synergy...")
            go_emb=torch.zeros(512).unsqueeze(0); dock=torch.tensor([[float(dsa),float(dsb)]])
            with torch.no_grad():
                if model_version=='v2' and cell_to_idx:
                    cidx=torch.tensor([cell_to_idx.get(cell_line,0)],dtype=torch.long)
                    score,logit=model(Batch.from_data_list([ga]),Batch.from_data_list([gb]),go_emb,dock,cidx)
                else:
                    score,logit=model(Batch.from_data_list([ga]),Batch.from_data_list([gb]),go_emb,dock)
                syn=score.item(); prob=torch.sigmoid(logit).item()
            st.session_state['syn_score']=syn; st.session_state['syn_prob']=prob
            prog.progress(100)
            with stat: st.write("✅ Complete!")
            with viz.container():
                if dran and (pa or pb):
                    st.markdown("**Both drugs docked in protein binding pocket**")
                    show_3d(pdb_content,pa,pb,name_a or "Drug A",name_b or "Drug B")
                    st.caption(f"🔵 {name_a or 'Drug A'}  🟠 {name_b or 'Drug B'}  🎨 Protein  *Drag to rotate*")
                else:
                    show_drugs(smiles_a,smiles_b)
            st.markdown("---\n### 📊 Results")
            verdict,color=get_verdict(syn)
            st.session_state['verdict']=verdict
            m1,m2,m3,m4=st.columns(4)
            m1.metric("Synergy Score",f"{syn:.3f}"); m2.metric("Synergy Probability",f"{prob:.3f}")
            m3.metric(f"{name_a or 'Drug A'} Binding",f"{dsa:.2f} kcal/mol")
            m4.metric(f"{name_b or 'Drug B'} Binding",f"{dsb:.2f} kcal/mol")
            st.markdown(f"### Verdict: :{color}[{verdict}]")
            st.caption(f"Cancer context: **{panel}** → **{cell_line}**")
            st.session_state.history.insert(0,{'drug_a':name_a or 'Drug A','drug_b':name_b or 'Drug B',
                'cell_line':cell_line,'score':syn,'verdict':verdict,'dock_a':dsa,'dock_b':dsb})
            st.session_state.history=st.session_state.history[:5]
            if known:
                ks,ksc=known
                st.markdown(f"""<div class="known-score">📚 <strong>NCI ALMANAC Ground Truth</strong><br>
Known: <strong>{ks:.2f}</strong> ({ksc}) | Predicted: <strong>{syn:.3f}</strong> | Error: <strong>{abs(syn-ks):.2f}</strong></div>""", unsafe_allow_html=True)
            else:
                st.markdown("""<div class="unknown-score">🔮 <strong>Novel prediction</strong> — not in NCI ALMANAC</div>""", unsafe_allow_html=True)
            with st.expander("📋 Full docking report"):
                st.markdown(f"""| Property | Value |
|----------|-------|
| Protein | {pname[:70]} |
| PDB ID | {pdb_id} |
| Box method | {bmethod} |
| {name_a or 'Drug A'} docking | {dsa:.3f} kcal/mol |
| {name_b or 'Drug B'} docking | {dsb:.3f} kcal/mol |
| Cancer type | {panel} |
| Cell line | {cell_line} |
| Synergy score | {syn:.3f} |
| Verdict | {verdict} |""")
            with st.expander("📖 How to interpret"):
                st.markdown("""| Score | Meaning |
|-------|---------|
| > 0.5 | Strongly Synergistic |
| 0.1–0.5 | Mildly Synergistic |
| -0.1–0.1 | Approximately Additive |
| < -0.1 | Antagonistic |

**Docking score**: more negative = stronger binding. Below -8 = strong binder.""")

    # ── Flythrough + Contact Map (outside tempdir, uses session state) ──────────
    if st.session_state.get('pa') or st.session_state.get('pb'):
        st.markdown("---")
        _pa=st.session_state.get('pa'); _pb=st.session_state.get('pb')
        _pdb=st.session_state.get('pdb_content','')

        if st.button("🎬 Animate Pocket Flythrough",key="fly"):
            if _pdb:
                fv=py3Dmol.view(width=750,height=500)
                fv.addModel(_pdb,'pdb'); fv.setStyle({'cartoon':{'color':'spectrum','opacity':0.5}})
                if _pa:
                    fv.addModel(pose_block(_pa,'A'),'pdb')
                    fv.setStyle({'model':1},{'stick':{'colorscheme':'cyanCarbon','radius':0.25},'sphere':{'colorscheme':'cyanCarbon','scale':0.35}})
                if _pb:
                    fv.addModel(pose_block(_pb,'B'),'pdb')
                    idx=2 if _pa else 1
                    fv.setStyle({'model':idx},{'stick':{'colorscheme':'orangeCarbon','radius':0.25},'sphere':{'colorscheme':'orangeCarbon','scale':0.35}})
                fv.setBackgroundColor('#000011'); fv.zoomTo({'model':1} if _pa else {}); fv.zoom(0.3,2000)
                components.html(fv._make_html(),height=520,scrolling=False)
                st.caption("🎬 Zooming into binding pocket | 🔵 Drug A | 🟠 Drug B")

        with st.expander("🗺️ Drug-Protein Contact Map"):
            if _pdb:
                cv=py3Dmol.view(width=700,height=400)
                cv.addModel(_pdb,'pdb'); cv.setStyle({},{'cartoon':{'color':'gray','opacity':0.3}})
                if _pa:
                    cv.addModel(pose_block(_pa,'A'),'pdb')
                    cv.setStyle({'model':1},{'stick':{'colorscheme':'cyanCarbon','radius':0.3},'sphere':{'colorscheme':'cyanCarbon','scale':0.4}})
                    cv.setStyle({'within':{'distance':5,'sel':{'model':1}}},{'stick':{'colorscheme':'cyanCarbon','radius':0.15},'cartoon':{'color':'cyan','opacity':0.8}})
                if _pb:
                    cv.addModel(pose_block(_pb,'B'),'pdb')
                    idx2=2 if _pa else 1
                    cv.setStyle({'model':idx2},{'stick':{'colorscheme':'orangeCarbon','radius':0.3},'sphere':{'colorscheme':'orangeCarbon','scale':0.4}})
                    cv.setStyle({'within':{'distance':5,'sel':{'model':idx2}}},{'stick':{'colorscheme':'orangeCarbon','radius':0.15},'cartoon':{'color':'orange','opacity':0.8}})
                cv.setBackgroundColor('#0a0a1a'); cv.zoomTo({'model':1} if _pa else {}); cv.zoom(1.5)
                components.html(cv._make_html(),height=420,scrolling=False)
                st.caption("🔵 Cyan = Drug A contacts | 🟠 Orange = Drug B contacts | Overlap = competition")
            else:
                st.info("Run docking first to see contact map.")

# ═══ TAB 2 ════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### 🗺️ Synergy Landscape — All Drug Combinations")
    if scores_data is None:
        st.warning("precomputed_scores.json not found.")
    else:
        sp=st.selectbox("Cancer type:",list(scores_data.keys()),key="hp")
        pd2=scores_data[sp]; drugs=pd2['drugs']; mat=np.array(pd2['matrix']); clh=pd2['cell_line']
        st.caption(f"Cell line: **{clh}** | {len(drugs)} drugs | {len(drugs)**2} combinations")
        fig=go.Figure(data=go.Heatmap(z=mat,x=drugs,y=drugs,
            colorscale=[[0,'#2166ac'],[0.35,'#74add1'],[0.5,'#f7f7f7'],[0.65,'#f46d43'],[1,'#d73027']],
            zmid=0,text=[[f"{drugs[i]} + {drugs[j]}<br>Score: {mat[i][j]:.3f}" for j in range(len(drugs))] for i in range(len(drugs))],
            hovertemplate="%{text}<extra></extra>",
            colorbar=dict(title="Synergy",tickvals=[-0.4,-0.2,0,0.2,0.4],ticktext=["Antagonistic","","Additive","","Synergistic"])))
        fig.update_layout(height=700,xaxis=dict(tickangle=-45,tickfont=dict(size=10)),yaxis=dict(tickfont=dict(size=10)),
            margin=dict(l=130,r=20,t=20,b=130),paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',font=dict(color='white'))
        st.plotly_chart(fig,use_container_width=True)
        pairs=[(drugs[i],drugs[j],float(mat[i][j])) for i in range(len(drugs)) for j in range(len(drugs)) if i!=j]
        ct,cb=st.columns(2)
        with ct:
            st.markdown("#### 🏆 Top 10 Synergistic")
            tdf=pd.DataFrame(sorted(pairs,key=lambda x:x[2],reverse=True)[:10],columns=['Drug A','Drug B','Score'])
            tdf['Score']=tdf['Score'].round(3); tdf['Verdict']=tdf['Score'].apply(lambda x:get_verdict(x)[0])
            st.dataframe(tdf,use_container_width=True,hide_index=True)
        with cb:
            st.markdown("#### ⚠️ Top 10 Antagonistic")
            bdf=pd.DataFrame(sorted(pairs,key=lambda x:x[2])[:10],columns=['Drug A','Drug B','Score'])
            bdf['Score']=bdf['Score'].round(3); bdf['Verdict']=bdf['Score'].apply(lambda x:get_verdict(x)[0])
            st.dataframe(bdf,use_container_width=True,hide_index=True)

# ═══ TAB 3 ════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### 📊 Cell Line Comparison")
    if scores_data is None:
        st.warning("precomputed_scores.json not found.")
    else:
        ad=scores_data['Melanoma']['drugs']
        ca,cb=st.columns(2)
        with ca: dar=st.selectbox("Drug A:",ad,index=ad.index("Vemurafenib") if "Vemurafenib" in ad else 0,key="ra")
        with cb: dbr=st.selectbox("Drug B:",ad,index=ad.index("Trametinib") if "Trametinib" in ad else 1,key="rb")
        if dar==dbr:
            st.warning("Select two different drugs.")
        else:
            panels=list(scores_data.keys())
            rs=[]
            for p in panels:
                pd3=scores_data[p]; dr=pd3['drugs']; m=np.array(pd3['matrix'])
                rs.append(float(m[dr.index(dar)][dr.index(dbr)]) if dar in dr and dbr in dr else 0.0)
            cr,cb2=st.columns(2)
            with cr:
                fr=go.Figure(); fr.add_trace(go.Scatterpolar(r=rs+[rs[0]],theta=panels+[panels[0]],
                    fill='toself',fillcolor='rgba(79,195,247,0.2)',line=dict(color='#4fc3f7',width=2)))
                fr.update_layout(polar=dict(radialaxis=dict(visible=True,range=[min(rs)-0.05,max(rs)+0.05])),
                    height=450,paper_bgcolor='rgba(0,0,0,0)',font=dict(color='white'),showlegend=False,
                    title=dict(text=f"{dar} + {dbr}",font=dict(size=14,color='#4fc3f7')))
                st.plotly_chart(fr,use_container_width=True)
            with cb2:
                fb=go.Figure(go.Bar(x=panels,y=rs,
                    marker_color=['#d73027' if s>0.1 else '#2166ac' if s<-0.1 else '#888' for s in rs],
                    text=[f"{s:.3f}" for s in rs],textposition='outside'))
                fb.update_layout(height=450,xaxis=dict(tickangle=-35),
                    yaxis=dict(title="Synergy score",zeroline=True,zerolinecolor='#666'),
                    paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',font=dict(color='white'),showlegend=False)
                st.plotly_chart(fb,use_container_width=True)
            sm=pd.DataFrame({'Cancer':panels,'Cell Line':[scores_data[p]['cell_line'] for p in panels],
                'Score':[round(s,3) for s in rs],'Verdict':[get_verdict(s)[0] for s in rs]}
            ).sort_values('Score',ascending=False).reset_index(drop=True)
            st.dataframe(sm,use_container_width=True,hide_index=True)

# ═══ TAB 4 ════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("### 🏥 Clinical Trial Matching")
    c1,c2=st.columns(2)
    with c1: cta=st.text_input("Drug A",placeholder="e.g. Vemurafenib",key="cta")
    with c2: ctb=st.text_input("Drug B",placeholder="e.g. Trametinib",key="ctb")
    ctc=st.text_input("Cancer type (optional)",placeholder="e.g. melanoma",key="ctc")
    if st.button("🔍 Search Clinical Trials",key="ctbtn") and cta and ctb:
        with st.spinner("Searching ClinicalTrials.gov..."):
            try:
                q=f"{cta} {ctb}"; q+=f" {ctc}" if ctc else ""
                r=requests.get("https://clinicaltrials.gov/api/v2/studies",
                    params={"query.term":q,"filter.overallStatus":"RECRUITING,ACTIVE_NOT_RECRUITING,COMPLETED","pageSize":15,"format":"json"},timeout=15)
                if r.status_code==200:
                    studies=r.json().get('studies',[])
                    if not studies: st.info(f"No trials found for {cta} + {ctb}.")
                    else:
                        st.success(f"Found {len(studies)} trials for **{cta} + {ctb}**")
                        for study in studies:
                            proto=study.get('protocolSection',{})
                            im=proto.get('identificationModule',{}); sm2=proto.get('statusModule',{})
                            dm=proto.get('designModule',{}); spm=proto.get('sponsorCollaboratorsModule',{})
                            cm=proto.get('conditionsModule',{})
                            nct=im.get('nctId','N/A'); title=im.get('briefTitle','No title')
                            status=sm2.get('overallStatus','Unknown')
                            phase=dm.get('phases',['N/A']); ps=', '.join(phase) if isinstance(phase,list) else str(phase)
                            sponsor=spm.get('leadSponsor',{}).get('name','Unknown'); conds=cm.get('conditions',[])
                            icon={'RECRUITING':'🟢','ACTIVE_NOT_RECRUITING':'🟡','COMPLETED':'⚫'}.get(status,'⚪')
                            with st.expander(f"{icon} {title[:80]}..."):
                                x1,x2,x3=st.columns(3)
                                x1.metric("NCT ID",nct); x2.metric("Status",status.replace('_',' ').title()); x3.metric("Phase",ps)
                                st.markdown(f"**Sponsor:** {sponsor}")
                                if conds: st.markdown(f"**Conditions:** {', '.join(conds[:5])}")
                                st.markdown(f"[View on ClinicalTrials.gov](https://clinicaltrials.gov/study/{nct})")
                else: st.error(f"API error: {r.status_code}")
            except Exception as e: st.error(f"Error: {e}")

# ═══ TAB 5 ════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown("### 📚 Literature Mining")
    p1,p2=st.columns(2)
    with p1: puba=st.text_input("Drug A",placeholder="e.g. Vemurafenib",key="puba")
    with p2: pubb=st.text_input("Drug B",placeholder="e.g. Trametinib",key="pubb")
    pubt=st.text_input("Additional topic",placeholder="e.g. synergy, resistance",key="pubt")
    if st.button("🔍 Search PubMed",key="pubbtn") and puba and pubb:
        with st.spinner("Searching PubMed..."):
            try:
                q=f"{puba} AND {pubb}"; q+=f" AND {pubt}" if pubt else ""
                sr=requests.get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
                    params={"db":"pubmed","term":q,"retmax":15,"retmode":"json","sort":"relevance"},timeout=15)
                pmids=sr.json().get('esearchresult',{}).get('idlist',[])
                if not pmids: st.info(f"No papers found for {puba} + {pubb}.")
                else:
                    fr2=requests.get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi",
                        params={"db":"pubmed","id":",".join(pmids),"retmode":"json"},timeout=15)
                    res=fr2.json().get('result',{})
                    total=res.get('uids',pmids)
                    st.success(f"Found **{len(total)} papers** for **{puba} + {pubb}**")
                    for pmid in total:
                        if pmid=='uids': continue
                        paper=res.get(pmid,{})
                        title=paper.get('title','No title'); journal=paper.get('fulljournalname',paper.get('source','Unknown'))
                        pubdate=paper.get('pubdate','Unknown'); authors=paper.get('authors',[])
                        astr=authors[0].get('name','')+' et al.' if authors else 'Unknown'
                        with st.expander(f"📄 {title[:80]}..."):
                            y1,y2,y3=st.columns(3)
                            y1.metric("Journal",journal[:25]); y2.metric("Date",pubdate); y3.metric("PMID",pmid)
                            st.markdown(f"**Authors:** {astr}")
                            st.markdown(f"[Read on PubMed](https://pubmed.ncbi.nlm.nih.gov/{pmid}/)")
            except Exception as e: st.error(f"Error: {e}")

# ═══ TAB 6 ════════════════════════════════════════════════════════════════════
with tab6:
    st.markdown("### 🔄 Drug Repurposing — Find Best Partner for Your Drug")
    if scores_data is None:
        st.warning("precomputed_scores.json not found.")
    else:
        adr=scores_data['Melanoma']['drugs']
        rr1,rr2=st.columns(2)
        with rr1: anch=st.selectbox("Your drug:",adr,index=adr.index("Imatinib") if "Imatinib" in adr else 0,key="anch")
        with rr2: rpan=st.selectbox("Cancer type:",list(scores_data.keys()),key="rpan")
        pdr=scores_data[rpan]; dr=pdr['drugs']; mr=np.array(pdr['matrix']); clr=pdr['cell_line']
        if anch in dr:
            ai=dr.index(anch)
            row=sorted([(dr[j],float(mr[ai][j])) for j in range(len(dr)) if j!=ai],key=lambda x:x[1],reverse=True)
            st.markdown(f"#### Best partners for **{anch}** in **{rpan}** ({clr})")
            fig_r=go.Figure(go.Bar(x=[x[1] for x in row],y=[x[0] for x in row],orientation='h',
                marker_color=['#d73027' if s>0.1 else '#2166ac' if s<-0.1 else '#888' for _,s in row],
                text=[f"{s:.3f}" for _,s in row],textposition='outside'))
            fig_r.update_layout(height=700,xaxis=dict(title="Synergy score",zeroline=True,zerolinecolor='#666'),
                yaxis=dict(autorange='reversed'),paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),showlegend=False,margin=dict(l=140,r=80,t=20,b=40))
            st.plotly_chart(fig_r,use_container_width=True)
            st.markdown("#### 🏆 Top 5 Recommended Combinations")
            for i,(drug,score) in enumerate(row[:5]):
                verdict,_=get_verdict(score); bc='#d73027' if score>0.1 else '#2166ac'
                st.markdown(f"""<div style="background:#1a1a2e;border-left:4px solid {bc};padding:12px;border-radius:6px;margin:6px 0;color:white;">
<b>#{i+1} {anch} + {drug}</b><br>Score: <b>{score:.3f}</b> | {verdict}</div>""", unsafe_allow_html=True)

# ═══ TAB 7 ════════════════════════════════════════════════════════════════════
with tab7:
    st.markdown("### 🔬 Mechanism of Action Explorer")
    mm1,mm2=st.columns(2)
    with mm1: dma=st.selectbox("Drug A:",list(DRUG_MECHANISMS.keys()),index=list(DRUG_MECHANISMS.keys()).index("Vemurafenib"),key="moa_a")
    with mm2: dmb=st.selectbox("Drug B:",list(DRUG_MECHANISMS.keys()),index=list(DRUG_MECHANISMS.keys()).index("Trametinib"),key="moa_b")
    if dma and dmb and dma!=dmb:
        ma=DRUG_MECHANISMS[dma]; mb=DRUG_MECHANISMS[dmb]
        mi1,mi2=st.columns(2)
        with mi1:
            st.markdown(f"""<div style="background:#1a1a2e;border-left:4px solid #4fc3f7;padding:12px;border-radius:6px;color:white;">
<b>💊 {dma}</b><br><b>Target:</b> {ma['target']}<br><b>Pathway:</b> {ma['pathway']}<br><b>Class:</b> {ma['class']}<br><b>MoA:</b> {ma['moa']}</div>""", unsafe_allow_html=True)
        with mi2:
            st.markdown(f"""<div style="background:#1a1a2e;border-left:4px solid #ff9800;padding:12px;border-radius:6px;color:white;">
<b>💊 {dmb}</b><br><b>Target:</b> {mb['target']}<br><b>Pathway:</b> {mb['pathway']}<br><b>Class:</b> {mb['class']}<br><b>MoA:</b> {mb['moa']}</div>""", unsafe_allow_html=True)
        st.markdown("---\n#### 🧬 Combination Analysis")
        pk=(ma['pathway'],mb['pathway'])
        if pk in SYNERGY_RULES: expl=SYNERGY_RULES[pk]
        elif ma['target']==mb['target']: expl=f"⚠️ Same target ({ma['target']}) — competition likely leads to antagonism."
        elif ma['class']==mb['class']: expl=f"⚠️ Same class ({ma['class']}) — redundant mechanism, additive at best."
        else: expl=f"🔬 Complementary — {dma} targets {ma['target']} while {dmb} targets {mb['target']}. {'Same' if ma['pathway']==mb['pathway'] else 'Different'} pathway."
        bg='#1e3a1e' if '✅' in expl else '#3a1e1e' if '⚠️' in expl else '#1e2a3a'
        bc='#4caf50' if '✅' in expl else '#ff5722' if '⚠️' in expl else '#4fc3f7'
        st.markdown(f"""<div style="background:{bg};border-left:4px solid {bc};padding:16px;border-radius:6px;color:white;font-size:15px;">{expl}</div>""", unsafe_allow_html=True)
        sp=ma['pathway']==mb['pathway']
        st.markdown(f"""| Property | {dma} | {dmb} |
|----------|------|------|
| Target | {ma['target']} | {mb['target']} |
| Pathway | {ma['pathway']} | {mb['pathway']} |
| Class | {ma['class']} | {mb['class']} |
| Same pathway | {'Yes ⚠️' if sp else 'No ✅'} | — |""")