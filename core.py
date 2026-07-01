"""
core.py — Pure business logic for ProteinSynergyDock, extracted from app.py
so it can be unit tested without importing Streamlit.

This module has ZERO dependency on streamlit. app.py imports everything
from here. Keeping logic here (not duplicated) means tests always exercise
the exact code path the deployed app runs.
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool
from rdkit import Chem
from rdkit.Chem import AllChem
from torch_geometric.data import Data
import os
import requests
import shutil
import subprocess

# ── Model definitions ──────────────────────────────────────────────────────────
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
    "Rucaparib":"CNCC1=CC=C(C=C1)C2=C3CCNC(=O)C4=CC(=CC(=C34)N2)F",
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
    "Alectinib":"CCC1=C(C=C2C(=C1)C(=O)C3=C(C2(C)C)NC4=C3C=CC(=C4)C#N)N5CCC(CC5)N6CCOCC6",
    "Afatinib":"CN(C)C/C=C/C(=O)Nc1cc2c(Nc3ccc(F)c(Cl)c3)ncnc2cc1OC",
    "Capecitabine":"CCOC(=O)Nc1nc(=O)n(C2OC(C)C(O)C2O)cc1F",
    "Temozolomide":"Cn1nnc2c(C(N)=O)ncn12",
    "Selumetinib":"Cc1cc(Nc2ncc(F)c(Nc3ccc(I)c(F)c3)n2)c(Cl)cc1Cl",
    "Belinostat":"O=C(/C=C/c1ccccc1)NOc1ccc(NS(=O)(=O)c2ccccc2)cc1",
    "Vorinostat":"O=C(CCCCCCC(=O)Nc1ccccc1)NO",
    "Crizotinib":"Cc1cn(C2CCNCC2)c2cc(Nc3ccc(F)cc3Cl)cnc12",
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
    "⚠️ Olaparib + Rucaparib (Mild Synergy)":{"smiles_a":"O=C1CCCN1c1ccc(cc1)C(=O)c1[nH]ncc1C1CC1","smiles_b":"CNCC1=CC=C(C=C1)C2=C3CCNC(=O)C4=CC(=CC(=C34)N2)F","pdb_id":"4DQY","name_a":"Olaparib","name_b":"Rucaparib","panel":"Ovarian Cancer","cell_line":"OVCAR-3","note":"PARP inhibition synergy. Known: **2.1**"},
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

MUTATION_DB = {
    "BRAF": {
        "wild_type": "7MNX",
        "mutations": {
            "V600E": {"pdb": "6PP9", "description": "Most common BRAF mutation (~50% melanomas). Vemurafenib targets this.", "drugs_affected": ["Vemurafenib", "Dabrafenib"]},
            "V600K": {"pdb": "6P7J", "description": "Second most common BRAF mutation. Reduced sensitivity to Vemurafenib.", "drugs_affected": ["Vemurafenib"]},
        }
    },
    "EGFR": {
        "wild_type": "1IVO",
        "mutations": {
            "T790M": {"pdb": "3UG2", "description": "Gatekeeper mutation — primary resistance to gefitinib/erlotinib.", "drugs_affected": ["Erlotinib", "Gefitinib", "Osimertinib"]},
            "L858R": {"pdb": "2ITX", "description": "Activating mutation — sensitizing, increases drug binding.", "drugs_affected": ["Erlotinib", "Gefitinib"]},
        }
    },
    "ALK": {
        "wild_type": "2XP2",
        "mutations": {
            "G1202R": {"pdb": "6MXM", "description": "Solvent-front mutation causing broad resistance to ALK inhibitors.", "drugs_affected": ["Crizotinib", "Alectinib"]},
            "L1196M": {"pdb": "4ANS", "description": "Gatekeeper mutation. Primary crizotinib resistance mechanism.", "drugs_affected": ["Crizotinib"]},
        }
    },
    "BCR-ABL": {
        "wild_type": "2HYY",
        "mutations": {
            "T315I": {"pdb": "2QOH", "description": "Gatekeeper mutation. Resistant to imatinib, dasatinib, nilotinib.", "drugs_affected": ["Imatinib", "Dasatinib", "Nilotinib"]},
            "E255K": {"pdb": "2HYY", "description": "P-loop mutation. Moderate resistance to imatinib.", "drugs_affected": ["Imatinib"]},
        }
    },
}

# ── Helper functions ───────────────────────────────────────────────────────────
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
        mol=Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol,AllChem.ETKDGv3())
        # Energy minimization — corrects bond lengths and angles from the
        # initial 3D embedding. MMFF94 is the standard force field for
        # small organic molecules; not doing this leaves the GNN operating
        # on geometrically unrealistic structures.
        result = AllChem.MMFFOptimizeMolecule(mol, maxIters=2000)
        if result == -1:  # MMFF setup failed, fall back to UFF
            AllChem.UFFOptimizeMolecule(mol, maxIters=2000)
        mol=Chem.RemoveHs(mol)
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
         '--exhaustiveness',str(exh),'--num_modes','9']
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

def get_verdict(s):
    if s>0.5: return "✅ Strongly Synergistic","green"
    elif s>0.1: return "⚠️ Mildly Synergistic","orange"
    elif s>-0.1: return "➖ Approximately Additive","blue"
    else: return "❌ Antagonistic","red"

def parse_nl_query(query, sc_data):
    q = query.lower().strip()
    CANCER_ALIASES = {
        "breast": ["MCF7", "MDA-MB-231", "T-47D"],
        "lung": ["A549", "NCI-H460", "HOP-92"],
        "leukemia": ["K-562", "CCRF-CEM", "HL-60"],
        "melanoma": ["UACC-62", "MALME-3M", "SK-MEL-5"],
        "colon": ["HCT-116", "HT29", "SW-620"],
        "ovarian": ["OVCAR-3", "IGROV1", "SK-OV-3"],
        "prostate": ["PC-3", "DU-145"],
        "cns": ["U251", "SF-268", "SF-295"],
        "renal": ["A498", "CAKI-1", "SN12C"],
    }
    DRUG_LIST = ["Vemurafenib", "Trametinib", "Erlotinib", "Imatinib", "Paclitaxel",
                 "Venetoclax", "Alpelisib", "Osimertinib", "Lapatinib", "Capecitabine",
                 "Palbociclib", "Ribociclib", "Dasatinib", "Crizotinib", "Dabrafenib"]
    if not sc_data:
        return "⚠️ No precomputed scores loaded. Visit the Synergy Landscape tab first."
    all_pairs = []
    for panel, drug_dict in sc_data.items():
        drugs = drug_dict.get('drugs', [])
        matrix = drug_dict.get('matrix', [])
        for i, d1 in enumerate(drugs):
            for j, d2 in enumerate(drugs):
                if i != j and matrix:
                    try:
                        all_pairs.append({"drug1": d1, "drug2": d2, "panel": panel, "score": float(matrix[i][j])})
                    except: pass
    if not all_pairs:
        return "⚠️ Could not parse scores. Check precomputed_scores.json format."
    df_all = pd.DataFrame(all_pairs)
    mentioned_drugs = [d for d in DRUG_LIST if d.lower() in q]
    detected_cancer = None
    for cancer in CANCER_ALIASES:
        if cancer in q:
            detected_cancer = cancer; break
    is_antagonistic = any(w in q for w in ["antagonistic", "antagonism", "worst", "avoid", "bad"])
    is_comparison = any(w in q for w in ["compare", "across", "different", "which cancer"])
    df_filtered = df_all.copy()
    if detected_cancer:
        cancer_lines = CANCER_ALIASES[detected_cancer]
        mask = df_filtered["panel"].apply(lambda p: any(cl.lower() in p.lower() for cl in cancer_lines))
        if mask.any():
            df_filtered = df_filtered[mask]
    if len(mentioned_drugs) == 2:
        d1, d2 = mentioned_drugs[0], mentioned_drugs[1]
        df_drug = df_filtered[
            ((df_filtered["drug1"]==d1)&(df_filtered["drug2"]==d2)) |
            ((df_filtered["drug1"]==d2)&(df_filtered["drug2"]==d1))
        ]
        if df_drug.empty:
            return f"❌ No data found for **{d1} + {d2}**. Try the Predict Synergy tab for real-time docking."
        avg_score = df_drug["score"].mean()
        label = "🟢 Synergistic" if avg_score > 0.1 else ("🔴 Antagonistic" if avg_score < -0.1 else "🟡 Additive")
        result = f"**{d1} + {d2}**: Average score = `{avg_score:.3f}` → **{label}**\n\n"
        for _, row in df_drug.sort_values("score", ascending=False).iterrows():
            result += f"- {row['panel']}: `{row['score']:.3f}`\n"
        return result
    elif len(mentioned_drugs) == 1:
        drug = mentioned_drugs[0]
        df_drug = df_filtered[(df_filtered["drug1"]==drug)|(df_filtered["drug2"]==drug)]
        if df_drug.empty:
            return f"❌ No data for **{drug}**."
        if is_comparison:
            result = f"**{drug}** across cancer types:\n\n"
            for panel, group in df_drug.groupby("panel"):
                avg = group["score"].mean()
                bar = "█" * max(1, int(abs(avg) * 10))
                result += f"- `{panel}`: {avg:+.3f} {bar}\n"
            return result
        partners = df_drug.copy()
        partners["partner"] = partners.apply(lambda r: r["drug2"] if r["drug1"]==drug else r["drug1"], axis=1)
        top = partners.groupby("partner")["score"].mean().sort_values(ascending=False)
        result = f"**Best combinations with {drug}**:\n\n"
        for partner, score in top.head(5).items():
            lbl = "🟢" if score > 0.1 else ("🔴" if score < -0.1 else "🟡")
            result += f"{lbl} **{drug} + {partner}**: `{score:.3f}`\n"
        return result
    else:
        if is_antagonistic:
            top = df_filtered.nsmallest(8, "score")
            result = f"**Most antagonistic pairs{' in ' + detected_cancer if detected_cancer else ''}:**\n\n"
            for _, row in top.iterrows():
                result += f"🔴 **{row['drug1']} + {row['drug2']}** (`{row['panel']}`): `{row['score']:.3f}`\n"
        else:
            top = df_filtered.nlargest(8, "score")
            result = f"**Most synergistic pairs{' in ' + detected_cancer if detected_cancer else ''}:**\n\n"
            for _, row in top.iterrows():
                result += f"🟢 **{row['drug1']} + {row['drug2']}** (`{row['panel']}`): `{row['score']:.3f}`\n"
        return result


def _enable_mc_dropout(model):
    """Sets only nn.Dropout submodules to train() mode (so they remain
    stochastic at inference time) while leaving every other layer
    (LayerNorm, Linear, GATv2Conv, etc.) in eval() mode.

    This is the key correctness detail of MC Dropout: naively calling
    model.train() on the whole model would also put any BatchNorm layers
    into training-statistics mode, which corrupts predictions. Since this
    architecture uses LayerNorm (not BatchNorm), that specific failure
    mode doesn't apply here, but selectively toggling only Dropout layers
    is the technically correct and architecture-agnostic approach, so
    this implementation stays correct even if BatchNorm is added later.
    """
    for module in model.modules():
        if module.__class__.__name__.startswith('Dropout'):
            module.train()


def predict_with_uncertainty(model, model_version, cell_to_idx, ga, gb, go_emb, dock,
                              cell_line, batch_cls, n_samples=20):
    """Runs n_samples stochastic forward passes with MC Dropout enabled and
    returns (mean_synergy, std_synergy, mean_prob, std_prob, all_synergy_samples).

    This converts a single point-estimate prediction into a distribution,
    giving a principled uncertainty estimate (Gal & Ghahramani, 2016 —
    dropout as a Bayesian approximation) without requiring model retraining
    or an ensemble of separately-trained models.

    Parameters mirror the existing single-shot inference call site in
    app.py exactly, so this is a drop-in replacement, not a new code path
    that has to be kept in sync with the original.
    """
    import torch as _torch

    model.eval()  # baseline: everything in eval mode...
    _enable_mc_dropout(model)  # ...then re-enable just the Dropout layers

    synergy_samples = []
    prob_samples = []

    with _torch.no_grad():
        for _ in range(n_samples):
            if model_version == 'v2' and cell_to_idx:
                cidx = _torch.tensor([cell_to_idx.get(cell_line, 0)], dtype=_torch.long)
                score, logit = model(batch_cls.from_data_list([ga]), batch_cls.from_data_list([gb]), go_emb, dock, cidx)
            else:
                score, logit = model(batch_cls.from_data_list([ga]), batch_cls.from_data_list([gb]), go_emb, dock)
            synergy_samples.append(score.item())
            prob_samples.append(_torch.sigmoid(logit).item())

    model.eval()  # restore full eval mode for any subsequent normal inference

    import numpy as _np
    synergy_samples = _np.array(synergy_samples)
    prob_samples = _np.array(prob_samples)

    return {
        "mean_synergy": float(synergy_samples.mean()),
        "std_synergy": float(synergy_samples.std()),
        "mean_prob": float(prob_samples.mean()),
        "std_prob": float(prob_samples.std()),
        "synergy_samples": synergy_samples.tolist(),
        "n_samples": n_samples,
    }


def confidence_label(std_synergy):
    """Maps prediction standard deviation to a human-readable confidence
    band. Thresholds are heuristic (like the existing get_verdict bands)
    and should be read as relative confidence, not a calibrated probability.
    """
    if std_synergy < 0.15:
        return "🟢 High confidence", "green"
    elif std_synergy < 0.4:
        return "🟡 Moderate confidence", "orange"
    else:
        return "🔴 Low confidence — model is uncertain about this pair", "red"
        

