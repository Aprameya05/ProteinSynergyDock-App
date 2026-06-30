"""
ProteinSynergyDockV2/V3 — Cell-Line-Held-Out Retraining (FAITHFUL VERSION)

This is built directly from your actual training notebook (moleculardocking.ipynb,
the self-contained cell that produced proteinsydock_v3_best.pt). Every line is
identical to your original run EXCEPT the train/val split logic, which is changed
from random row-shuffling to holding out entire cell lines.

WHY THIS MATTERS: your original model was evaluated on a RANDOM split — every
cell line appears in both train and test, just not the exact same drug-pair
rows. This is the easiest possible evaluation. This script instead holds out
~20% of cell lines ENTIRELY (never seen in training at all), which tests
whether the model generalizes to a cancer type it's never seen synergy data
for. This is the harder, more clinically meaningful test.

EXPECTED OUTCOME: held-out r/AUROC will likely be LOWER than 0.5667/0.7946.
This is normal, expected, and is itself the honest finding — report both
numbers together, and the gap between them.

USAGE:
1. In a new Kaggle notebook, attach the same datasets as your original run:
   - aprameyabharadwaj111/nci-almanac
   - aprameyabharadwaj111/proteinsydock-data
   - aprameyabharadwaj111/proteinsydock-docking-v2
2. Run this entire script as one cell (or split into logical chunks, doesn't matter)
3. Wait for training to complete (~100 epochs, similar time to your original run)
4. At the end, download heldout_results.json from /kaggle/working/ and share it back

IMPORTANT: removed save_checkpoint_to_kaggle() call since that function is
specific to your notebook environment setup (likely a custom helper for
auto-saving to a Kaggle dataset) and wasn't visible in the cell I extracted
from. If you have that helper defined elsewhere in your notebook, you can
add the call back in after the torch.save() line. It's not required for
this experiment to work — it would just add convenience auto-backup.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool
from torch_geometric.data import Data, Batch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm
import json, os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
WORK_DIR = '/kaggle/working'
DATA_DIR = '/kaggle/input/datasets/aprameyabharadwaj111/proteinsydock-data'
ALMANAC  = '/kaggle/input/datasets/aprameyabharadwaj111/nci-almanac/ComboDrugGrowth_Nov2017.csv'

DRUG_SMILES = {
    "Imatinib":"CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)NC4=NC=CC(=N4)C5=CN=CC=C5",
    "Gefitinib":"COC1=C(C=C2C(=C1)N=CN=C2NC3=CC(=C(C=C3)F)Cl)OCCCN4CCOCC4",
    "Erlotinib":"COCCOC1=C(C=C2C(=C1)C(=NC=N2)NC3=CC=CC(=C3)C#C)OCCOC",
    "Lapatinib":"CS(=O)(=O)CCNCc1oc(cc1)c2ccc3ncnc(Nc4ccc(Oc5cccc(Cl)c5)c(Cl)c4)c3c2",
    "Dasatinib":"Cc1nc(Nc2ncc(s2)C(=O)Nc2c(C)cccc2Cl)cc(n1)N1CCN(CCO)CC1",
    "Nilotinib":"Cc1cn(c2cc(NC(=O)c3ccc(C)c(Nc4nccc(n4)-c4cccnc4)c3)cc(C(F)(F)F)c12)C",
    "Sorafenib":"CNC(=O)c1cc(Oc2ccc(NC(=O)Nc3ccc(Cl)c(C(F)(F)F)c3)cc2)ccn1",
    "Vemurafenib":"CCCS(=O)(=O)Nc1ccc(F)c(C(=O)c2c[nH]c3ncc(-c4ccc(Cl)cc4)cc23)c1",
    "Dabrafenib":"CC(C)(C)c1nc2cc(F)ccc2c(C(=O)Nc2ccc(F)c(NS(=O)(=O)c3ccc(F)cc3)c2)n1",
    "Trametinib":"CC(=O)Nc1ccc(-c2cc3c(nc(N)nc3n2C)N2CCC(F)(F)CC2=O)cc1F",
    "Cobimetinib":"OC(COc1cc(Cl)c(F)cc1F)CN1CCC(=C1)c1cc2c(Nc3ccc(F)cc3F)ncc(C(N)=O)c2[nH]1",
    "Sunitinib":"CCN(CC)CCNC(=O)c1c(C)[nH]c(C=C2C(=O)Nc3ccc(F)cc32)c1C",
    "Axitinib":"CNC(=O)c1ccc(cc1)Oc1cccc(c1)N1C(=O)/C(=C/c2ccc(s2)NC(=O)/C=C/c2ccc(Cl)cc2)/CC1",
    "Everolimus":"CCC(CC)COC(=O)C1CC(CC(=O)O1)CC(CC(=O)O)C(C)CC=CC(C)C(O)C",
    "Temsirolimus":"CCC(CC)COC(=O)C1CC(CC(=O)O1)CC(CC(=O)O)C(C)CC=CC(C)C(O)C",
    "Vorinostat":"O=C(CCCCCCC(=O)Nc1ccccc1)NO",
    "Ibrutinib":"C=CC(=O)N1CCCC(c2ncnc3[nH]ccc23)C1",
    "Venetoclax":"CC1(CCC(CC1)N2CCN(CC2)c3ccc(cc3)C(=O)NS(=O)(=O)c4ccc(cc4-c5cnc6ccccc6n5)Cl)C",
    "Olaparib":"O=C1CCCN1c1ccc(cc1)C(=O)c1[nH]ncc1C1CC1",
    "Niraparib":"OC(=O)c1ccc2[nH]ncc2c1-c1ccc(cn1)C1CCNCC1",
    "Rucaparib":"NCc1cc2cc(F)ccc2[nH]1-c1ccc3NCCCC(=O)c3c1",
    "Palbociclib":"CC1=C(C(=NC(=C1)N2CCNCC2)N3CCCC3)C(=O)NC4=CC=CC=N4",
    "Abemaciclib":"CC1=NC(=NC(=C1)NC2=NC=CC(=N2)N3CCC(CC3)NC(=O)C4=CC=C(C=C4)F)C5=CC(=CC=C5)F",
    "Ribociclib":"CC1=NC(=NC(=C1)N2CCNCC2)C3=CC4=C(C=C3)N=CN=C4N5CCCC5",
    "Ponatinib":"Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1C#Cc1cnc2ccccc12",
    "Ruxolitinib":"CC(CC#N)n1cc(-c2ncnc3[nH]ccc23)cn1",
    "Paclitaxel":"O=C(OC1C[C@]2(O)C(=O)C(OC(=O)c3ccccc3)C(O)C(OC(=O)C(NC(=O)c3ccccc3)c3ccccc3)C2(C)CC1)C(C)=C",
    "Docetaxel":"CC(C)(C)OC(=O)NC(c1ccccc1)C(O)C(=O)OC1CC2OCC2(OC(C)=O)C1OC(=O)c1ccccc1",
    "Doxorubicin":"COc1cccc2C(=O)c3c(O)c4CC(O)(CC(OC5CC(N)C(O)C(C)O5)c4c(O)c3C(=O)c12)C(=O)CO",
    "Etoposide":"COc1cc2c(cc1OC)CC1C(=O)OCC1(c1cc3c(cc1OC)OCO3)C2",
    "Irinotecan":"O=C(OCC1=C2CN3CCC4=CC5=CC(=O)OC5=NC4=C3C2=NC1=O)c1cccnc1",
    "Topotecan":"OCC1=C2CN3CCC4=CC5=CC(=O)OC5=NC4=C3C2=NC1=O",
    "Methotrexate":"CN(c1ccc(cc1)C(=O)NC(CCC(=O)O)C(=O)O)c1nc(N)nc2ccc(CNC3=CC=CC=C3)cc12",
    "Pemetrexed":"CN1CC2=CC=C(C=C2N=C1C(=O)NC(CCC(=O)O)C(=O)O)C(=O)O",
    "Gemcitabine":"NC(=O)C1=CN(C(=O)N1)C1CC(F)(F)C(CO)O1",
    "Cytarabine":"Nc1ccn(C2OC(CO)C(O)C2O)c(=O)n1",
    "Fludarabine":"Nc1nc2n(cnc2c(=O)[nH]1)C1OC(CO)C(O)C1F",
    "Cyclophosphamide":"ClCCN(CCCl)P(=O)(N)OCC1CCCCO1",
    "Melphalan":"NC(=O)c1ccc(N(CCCl)CCCl)cc1",
    "Carmustine":"ClCCNC(=O)N(N=O)CCCl",
    "Lomustine":"ClCCN(C(=O)N(N=O)CCCl)C1CCCCC1",
    "Vincristine":"CCC1(CC(CC2(C1N(C)c3ccccc23)CCO)OC(=O)C(C(CC4=CC5=C(CN(C)C4)C6=CC=CC=C56)(C(=O)OC)O)NC(=O)OC)O",
    "Vinblastine":"CCC1(CC(CC2(C1N(C)c3ccccc23)CCO)OC(=O)C(C(CC4=CC5=C(CN(C)C4)C6=CC=CC=C56)(C(=O)OC)O)NC(=O)OC)O",
    "Mercaptopurine":"Sc1ncnc2[nH]cnc12",
    "Thioguanine":"Nc1nc2[nH]cnc2c(=O)[nH]1",
    "Capecitabine":"CCOC(=O)Nc1nc(=O)n(C2OC(C)C(O)C2O)cc1F",
    "Temozolomide":"Cn1nnc2c(C(N)=O)ncn12",
    "Selumetinib":"Cc1cc(Nc2ncc(F)c(Nc3ccc(I)c(F)c3)n2)c(Cl)cc1Cl",
    "Osimertinib":"C=CC(=O)Nc1cc2c(Nc3ccc(F)c(Cl)c3)nc(OC)nc2cc1N(C)CCN(C)C",
    "Alectinib":"COc1cc2c(cc1N1CCC(CC1)c1ccc3[nH]ccc3c1)cc(=O)n1ccc(C#N)c21",
    "Dactinomycin":"CC1OC(=O)C(N(C)C(=O)c2c(N)c(Cl)c(=O)c3nccc23)C(=O)N(C)CC(=O)N(C)C1",
    "Hydroxyurea":"NC(=O)NO",
    "Vismodegib":"Clc1ccc(S(=O)(=O)C2=CC(=O)NC(=O)N2)cc1Cl",
    "Bicalutamide":"CC(CS(=O)(=O)c1ccc(F)cc1)(C(=O)Nc1ccc(cc1)[N+](=O)[O-])C(F)(F)F",
    "Flutamide":"CC(C(=O)Nc1ccc(cc1)[N+](=O)[O-])C(F)(F)F",
    "Cabozantinib":"COc1cc2nccc(Oc3ccc(NC(=O)C4(C(=O)Nc5ccc(F)cc5)CC4)cc3)c2cc1OC",
    "Afatinib":"CN(C)C/C=C/C(=O)Nc1cc2c(Nc3ccc(F)c(Cl)c3)ncnc2cc1OC",
    "Chlorambucil":"OC(=O)CCCc1ccc(N(CCCl)CCCl)cc1",
    "Vinorelbine":"CCC1(CC(=O)OC)C=C2CN3CCc4cc5c(cc4C3C2CC1)(OCO5)CC(=O)OC",
    "Mitoxantrone":"O=C1c2cccc(NCCNCCO)c2C(=O)c2c1ccc(NCCNCCO)c2O",
    "Aminopterin":"Nc1nc2ncc(CNc3ccc(cc3)C(=O)NC(CCC(=O)O)C(=O)O)nc2c(=O)[nH]1",
    "Ixazomib":"CC(CC(=O)N1CC(=O)NC1=O)NC(=O)c1cc(Cl)cc(Cl)c1",
    "Enzalutamide":"CN1C(=O)C(c2cccc(C#N)c2)(CC1=O)c1ccc(cc1)[N+](=O)[O-]",
    "Dacarbazine":"CN(C)N=Nc1c(C(N)=O)ncn1C",
    "Teniposide":"O=C1OCC2(c3cc4c(cc3OC3OC(CO)C(O)C(O)C3O)OCO4)OC3C(O)C(O)C(CO)OC3C12",
    "Belinostat":"O=C(/C=C/c1ccccc1)NOc1ccc(NS(=O)(=O)c2ccccc2)cc1",
    "Panobinostat":"CC/C=C/C(=O)NOc1ccc(CNc2ccc3cccc(c3c2)/C=C/c2cccnc2)cc1",
    "Romidepsin":"CC(C)/C=C(/C)C(=O)NC(C/C=C/CSS1)C(=O)NC(Cc2ccccc2)C(=O)N(C)C(C(C)C)C(=O)NC1CC(C)C",
    "Copanlisib":"Cc1cn2c(n1)CC(Nc1nc(Nc3cccc(S(N)(=O)=O)c3)ncc1F)CC2",
    "Duvelisib":"CC1CN(c2nc(Nc3cccc(Cl)c3)c3[nH]ncc3n2)CC(C)O1",
    "Alpelisib":"CC1(C)CN(c2nc(Nc3ccc(S(N)(=O)=O)cc3F)ncc2F)CC1=O",
    "Brigatinib":"CNC(=O)c1cc(Oc2ccc(NC3=NC(=CC(=N3)N3CCC(N4CCN(C)CC4)CC3)Cl)cc2P(C)(C)=O)ccn1",
    "Lorlatinib":"CC#Cc1ccc2c(c1)N(C)C(=O)c1cc(F)cnc1NC(=O)O2",
    "Tucatinib":"Cc1nc(Nc2ccc(Oc3ccnc(NC4CC4)n3)cc2)cc(N2CCN(C(=O)c3ccco3)CC2)n1",
    "Selinexor":"O=C(/C=C/c1ccc2cc(F)ccc2c1)N/N=C/c1ccncc1",
    "Zanubrutinib":"O=C(/C=C/c1ccco1)N1CCC(n2nc(-c3ccc4c(c3)CCNC4=O)c3c(N)ncnc23)CC1",
    "Acalabrutinib":"CC#CC(=O)N1CCC(n2nc(-c3ccc4c(c3)CCNC4=O)c3c(N)ncnc23)CC1",
    "Mitomycin C":"C[C@@]1(OC(N)=O)[C@@H]2CC(=C(CN)N2C1=O)OC",
    "Bleomycin":"NCC(=O)NC(C(=O)NC(=O)NC(C(=O)NC(C(=O)N)CCCNC(=N)N)CSC)C(O)C",
    "Streptozotocin":"OCC1OC(NC(=O)N=O)(n2cncn2)C(O)C1O",
    "Selpercatinib":"CCc1cc(Nc2ncc(C#N)c(-c3cn(C4CCNCC4)c4ncccc34)n2)ccc1OC",
}

DRUG_TO_PDB = {
    "Imatinib":"2HYY","Gefitinib":"1IVO","Erlotinib":"1IVO","Lapatinib":"1IVO",
    "Dasatinib":"2HYY","Nilotinib":"2HYY","Sorafenib":"3OG7","Vemurafenib":"3OG7",
    "Dabrafenib":"3OG7","Trametinib":"3EQH","Cobimetinib":"3EQH","Sunitinib":"3MJG",
    "Axitinib":"3MJG","Everolimus":"1FAP","Temsirolimus":"1FAP","Vorinostat":"4BKX",
    "Ibrutinib":"3K54","Venetoclax":"4LVT","Olaparib":"4DQY","Niraparib":"4DQY",
    "Rucaparib":"4DQY","Palbociclib":"2W96","Abemaciclib":"2W96","Ribociclib":"2W96",
    "Ponatinib":"2HYY","Ruxolitinib":"3LXK","Paclitaxel":"1JFF","Docetaxel":"1JFF",
    "Vincristine":"1JFF","Vinblastine":"1JFF","Doxorubicin":"1ZXM","Etoposide":"1ZXM",
    "Irinotecan":"1K4T","Topotecan":"1K4T","Methotrexate":"1DHF","Pemetrexed":"1DHF",
    "Gemcitabine":"2BPE","Cytarabine":"2BPE","Fludarabine":"2BPE",
    "Cyclophosphamide":"1SUK","Melphalan":"1SUK","Carmustine":"1RZ0","Lomustine":"1RZ0",
    "Mercaptopurine":"1DHF","Thioguanine":"1DHF","Capecitabine":"1HVY",
    "Temozolomide":"1RZ0","Selumetinib":"3EQH","Osimertinib":"1IVO",
    "Alectinib":"2XP2","Dactinomycin":"1ZXM","Hydroxyurea":"2BPE",
    "Vismodegib":"1RZ0","Bicalutamide":"1RZ0","Flutamide":"1RZ0",
    "Cabozantinib":"3MJG","Afatinib":"1IVO","Chlorambucil":"1SUK",
    "Vinorelbine":"1JFF","Mitoxantrone":"1ZXM","Aminopterin":"1DHF",
    "Ixazomib":"2F16","Enzalutamide":"1RZ0","Dacarbazine":"1RZ0",
    "Teniposide":"1ZXM","Belinostat":"4BKX","Panobinostat":"4BKX",
    "Romidepsin":"4BKX","Mitomycin C":"1SUK","Bleomycin":"1ZXM",
    "Streptozotocin":"1RZ0","Copanlisib":"2RD0","Duvelisib":"2RD0",
    "Alpelisib":"2RD0","Brigatinib":"2XP2","Lorlatinib":"2XP2",
    "Tucatinib":"1IVO","Selinexor":"1RZ0","Zanubrutinib":"3K54",
    "Acalabrutinib":"3K54","Selpercatinib":"3MJG",
}

DRUG_TO_UNIPROT = {
    "Imatinib":"P00519","Gefitinib":"P00533","Erlotinib":"P00533","Lapatinib":"P00533",
    "Dasatinib":"P00519","Nilotinib":"P00519","Sorafenib":"P15056","Vemurafenib":"P15056",
    "Dabrafenib":"P15056","Trametinib":"Q02750","Cobimetinib":"Q02750","Sunitinib":"P09619",
    "Axitinib":"P35968","Everolimus":"P42345","Temsirolimus":"P42345","Vorinostat":"Q13547",
    "Ibrutinib":"Q06187","Venetoclax":"Q07817","Olaparib":"P09874","Niraparib":"P09874",
    "Rucaparib":"P09874","Palbociclib":"P11802","Abemaciclib":"P11802","Ribociclib":"P11802",
    "Ponatinib":"P00519","Ruxolitinib":"P23458","Paclitaxel":"P07437","Docetaxel":"P07437",
    "Vincristine":"P07437","Vinblastine":"P07437","Doxorubicin":"P11388","Etoposide":"P11388",
    "Irinotecan":"P11387","Topotecan":"P11387","Methotrexate":"P00374","Pemetrexed":"P00374",
    "Gemcitabine":"P23921","Cytarabine":"P23921","Fludarabine":"P23921",
    "Cyclophosphamide":"P11166","Melphalan":"P11166","Carmustine":"P09884","Lomustine":"P09884",
    "Mercaptopurine":"P00374","Thioguanine":"P00374","Capecitabine":"P48547",
    "Temozolomide":"P09884","Selumetinib":"Q02750","Osimertinib":"P00533",
    "Alectinib":"Q9UM73","Dactinomycin":"P11388","Hydroxyurea":"P23921",
    "Vismodegib":"P09884","Bicalutamide":"P09884","Flutamide":"P09884",
    "Cabozantinib":"P09619","Afatinib":"P00533","Chlorambucil":"P11166",
    "Vinorelbine":"P07437","Mitoxantrone":"P11388","Aminopterin":"P00374",
    "Ixazomib":"P28070","Enzalutamide":"P09884","Dacarbazine":"P09884",
    "Teniposide":"P11388","Belinostat":"Q13547","Panobinostat":"Q13547",
    "Romidepsin":"Q13547","Mitomycin C":"P11166","Bleomycin":"P11388",
    "Streptozotocin":"P09884","Copanlisib":"P42336","Duvelisib":"P42336",
    "Alpelisib":"P42336","Brigatinib":"Q9UM73","Lorlatinib":"Q9UM73",
    "Tucatinib":"P00533","Selinexor":"P09884","Zanubrutinib":"Q06187",
    "Acalabrutinib":"Q06187","Selpercatinib":"P09619",
}

NSC_TO_DRUG = {
    752:"Methotrexate",755:"Mercaptopurine",762:"Thioguanine",3088:"Melphalan",
    49842:"Vincristine",67574:"Cytarabine",79037:"Cisplatin",102816:"Cyclophosphamide",
    119875:"Oxaliplatin",122758:"Etoposide",123127:"Doxorubicin",125066:"Paclitaxel",
    127716:"Carboplatin",141540:"Docetaxel",148832:"Irinotecan",169780:"Topotecan",
    218321:"Gemcitabine",226080:"Fludarabine",266046:"Imatinib",282388:"Bortezomib",
    330507:"Sorafenib",332488:"Sunitinib",362856:"Erlotinib",380265:"Lapatinib",
    609699:"Pemetrexed",613327:"Dasatinib",624152:"Nilotinib",630176:"Vorinostat",
    637793:"Temsirolimus",639829:"Everolimus",643356:"Gefitinib",666056:"Crizotinib",
    683864:"Vemurafenib",700499:"Ruxolitinib",701852:"Axitinib",704735:"Carfilzomib",
    710407:"Ibrutinib",713563:"Idelalisib",716190:"Ponatinib",718781:"Trametinib",
    719276:"Dabrafenib",724770:"Palbociclib",726992:"Olaparib",729971:"Venetoclax",
    732517:"Cobimetinib",747599:"Abemaciclib",761431:"Ribociclib",763371:"Niraparib",
    773990:"Rucaparib",740:"Aminopterin",3053:"Dactinomycin",8806:"Dactinomycin",
    24559:"Streptozotocin",26980:"Mitomycin C",32065:"Chlorambucil",38721:"Hydroxyurea",
    45388:"Dacarbazine",77213:"Bleomycin",118218:"Flutamide",122819:"Teniposide",
    125973:"Mitoxantrone",138783:"Vinorelbine",241240:"Bicalutamide",256439:"Capecitabine",
    296961:"Belinostat",369100:"Panobinostat",409962:"Romidepsin",606869:"Temozolomide",
    628503:"Enzalutamide",673596:"Vismodegib",698037:"Copanlisib",702294:"Duvelisib",
    712807:"Cabozantinib",715055:"Alpelisib",719344:"Osimertinib",719345:"Osimertinib",
    719627:"Alectinib",721517:"Brigatinib",737754:"Lorlatinib",743414:"Afatinib",
    747971:"Selpercatinib",749226:"Tucatinib",750690:"Ixazomib",753082:"Selinexor",
    754230:"Zanubrutinib",755986:"Acalabrutinib",756645:"Selumetinib",
}

PANEL_TO_IDX = {
    'Renal Cancer':0,'Melanoma':1,'Non-Small Cell Lung Cancer':2,
    'Colon Cancer':3,'Leukemia':4,'Breast Cancer':5,
    'Ovarian Cancer':6,'CNS Cancer':7,'Prostate Cancer':8,'Unknown':9
}

# ── Build dataset ─────────────────────────────────────────────────────────────

print("Loading ALMANAC...")
df_raw = pd.read_csv(ALMANAC, usecols=['NSC1','NSC2','SCORE','CELLNAME','PANEL','VALID'])
df_raw = df_raw[df_raw['VALID']=='Y'].copy()
df_raw['drug_a'] = df_raw['NSC1'].map(NSC_TO_DRUG)
df_raw['drug_b'] = df_raw['NSC2'].map(NSC_TO_DRUG)
df_raw = df_raw.dropna(subset=['drug_a','drug_b'])

df_cell = df_raw.groupby(['drug_a','drug_b','CELLNAME','PANEL'])['SCORE'].agg(
    ['mean','count']).reset_index()
df_cell.columns = ['drug_a','drug_b','cell_line','panel','synergy','n_measurements']
df_cell = df_cell[df_cell['n_measurements']>=2]

rows = []
for _, row in df_cell.iterrows():
    da,db = row['drug_a'],row['drug_b']
    if da not in DRUG_SMILES or db not in DRUG_SMILES: continue
    if da not in DRUG_TO_PDB or db not in DRUG_TO_PDB: continue
    rows.append({
        'drug_a':da,'drug_b':db,
        'smiles_a':DRUG_SMILES[da],'smiles_b':DRUG_SMILES[db],
        'pdb_id':DRUG_TO_PDB[da],'uniprot':DRUG_TO_UNIPROT.get(da,'unknown'),
        'cell_line':row['cell_line'],'panel':row['panel'],
        'panel_idx':PANEL_TO_IDX.get(row['panel'],9),
        'synergy':round(row['synergy'],3),
        'synergy_class':int(row['synergy']>2.0),
        'n_measurements':int(row['n_measurements']),
    })

df_full = pd.DataFrame(rows)
df_full = df_full[df_full['synergy'].between(-30,30)].reset_index(drop=True)

# Add docking scores
dock_path = '/kaggle/input/datasets/aprameyabharadwaj111/proteinsydock-docking-v2/docking_results_full.json'
with open(dock_path) as f:
    docking_results = json.load(f)

def get_dock(drug, pdb_id):
    return docking_results.get(f"{drug}_{pdb_id}",{}).get('score',-7.0)

df_full['dock_score_a'] = df_full.apply(lambda r: get_dock(r['drug_a'],r['pdb_id']), axis=1)
df_full['dock_score_b'] = df_full.apply(lambda r: get_dock(r['drug_b'],r['pdb_id']), axis=1)

# Normalize
scaler = StandardScaler()
df_full['synergy_scaled'] = scaler.fit_transform(df_full[['synergy']])

print(f"Dataset: {len(df_full):,} triplets | {df_full[['drug_a','drug_b']].drop_duplicates().shape[0]} pairs")

# ── CELL-LINE-HELD-OUT SPLIT (instead of random) ────────────────────────────
# This is the ONLY change from the original training run. Instead of
# shuffling all rows randomly (which lets every cell line appear in both
# train and val), we hold out entire cell lines. The model NEVER sees any
# synergy data for held-out cell lines during training, and is only
# evaluated on them at the very end. This tests true generalization to
# unseen cancer types, which the original random split did not test.
np.random.seed(42)

all_cell_lines_full = sorted(df_full['cell_line'].unique())
n_total_cl = len(all_cell_lines_full)
N_HELDOUT = max(1, round(n_total_cl * 0.20))
heldout_cell_lines = set(np.random.choice(all_cell_lines_full, size=N_HELDOUT, replace=False))
train_cell_lines = set(all_cell_lines_full) - heldout_cell_lines

print(f"Total cell lines: {n_total_cl}")
print(f"Held out ({len(heldout_cell_lines)}): {sorted(heldout_cell_lines)}")
print(f"Training on ({len(train_cell_lines)}): {sorted(train_cell_lines)}")

train_full_df = df_full[df_full['cell_line'].isin(train_cell_lines)].reset_index(drop=True)
heldout_df = df_full[df_full['cell_line'].isin(heldout_cell_lines)].reset_index(drop=True)

# Within the training cell lines, do a random 90/10 train/val split for
# model selection (early stopping) — this is fine since it doesn't touch
# the held-out cell lines at all.
val_frac = 0.10
val_mask = np.random.rand(len(train_full_df)) < val_frac
val_df = train_full_df[val_mask].reset_index(drop=True)
train_df = train_full_df[~val_mask].reset_index(drop=True)

print(f"Train: {len(train_df):,} | Val: {len(val_df):,} | Held-out test: {len(heldout_df):,} (CELL-LINE-HELD-OUT SPLIT)")

assert len(train_cell_lines & heldout_cell_lines) == 0, "BUG: overlap between train and held-out cell lines!"
print("✅ Confirmed zero overlap between training and held-out cell lines.")

# cell_line_to_idx must cover ALL cell lines including held-out ones, so the
# embedding layer has a valid (if undertrained) index for them at test time
cell_lines = sorted(df_full['cell_line'].unique())
cell_line_to_idx = {cl:i for i,cl in enumerate(cell_lines)}
n_cell_lines = len(cell_lines)
go_ctx = torch.load(f'{DATA_DIR}/go_context_embeddings.pt', weights_only=False)
for k in go_ctx:
    if go_ctx[k].dim()>1: go_ctx[k]=go_ctx[k].squeeze(0)

# ── Model ─────────────────────────────────────────────────────────────────────

class DrugEncoder(nn.Module):
    def __init__(self,in_dim=7,hidden=128,out_dim=256,heads=4):
        super().__init__()
        self.proj=nn.Linear(in_dim,hidden)
        self.conv1=GATv2Conv(hidden,hidden,heads=heads,concat=True)
        self.conv2=GATv2Conv(hidden*heads,out_dim,heads=1,concat=False)
        self.norm1=nn.LayerNorm(hidden*heads); self.norm2=nn.LayerNorm(out_dim)
    def forward(self,x,edge_index,batch):
        x=F.gelu(self.proj(x))
        x=F.gelu(self.norm1(self.conv1(x,edge_index)))
        x=F.gelu(self.norm2(self.conv2(x,edge_index)))
        return global_mean_pool(x,batch)

class CrossDrugAttention(nn.Module):
    def __init__(self,dim=256):
        super().__init__()
        self.attn=nn.MultiheadAttention(dim,num_heads=4,batch_first=True)
        self.norm=nn.LayerNorm(dim); self.ff=nn.Sequential(nn.Linear(dim,dim*2),nn.GELU(),nn.Linear(dim*2,dim))
    def forward(self,a,b):
        seq=torch.stack([a,b],dim=1); att,_=self.attn(seq,seq,seq)
        seq=self.norm(seq+att); seq=seq+self.ff(seq)
        return seq.reshape(seq.shape[0],-1)

class ProteinSynergyDockV2(nn.Module):
    def __init__(self,go_dim=512,drug_dim=256,hidden=512,n_cell_lines=60):
        super().__init__()
        self.drug_encoder=DrugEncoder(in_dim=7,hidden=128,out_dim=drug_dim)
        self.cross_attn=CrossDrugAttention(dim=drug_dim)
        self.film_scale=nn.Linear(go_dim,drug_dim*2)
        self.film_bias=nn.Linear(go_dim,drug_dim*2)
        self.cell_embed=nn.Embedding(n_cell_lines,32)
        self.head=nn.Sequential(
            nn.Linear(drug_dim*2+2+32,hidden),nn.LayerNorm(hidden),nn.ReLU(),nn.Dropout(0.2),
            nn.Linear(hidden,hidden//2),nn.ReLU(),nn.Dropout(0.1),nn.Linear(hidden//2,2))
    def forward(self,da,db,go,dock,cell_idx):
        ea=self.drug_encoder(da.x,da.edge_index,da.batch)
        eb=self.drug_encoder(db.x,db.edge_index,db.batch)
        fused=self.cross_attn(ea,eb)
        fused=fused*(1+self.film_scale(go))+self.film_bias(go)
        cell=self.cell_embed(cell_idx)
        fused=torch.cat([fused,dock,cell],dim=-1)
        out=self.head(fused)
        return out[:,0],out[:,1]

def smiles_to_graph(smiles):
    mol=Chem.MolFromSmiles(smiles)
    if mol is None: return None
    try:
        mol=Chem.AddHs(mol); AllChem.EmbedMolecule(mol,AllChem.ETKDGv3()); mol=Chem.RemoveHs(mol)
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

def pearson_r(x,y):
    x=x-x.mean(); y=y-y.mean()
    return (x*y).sum()/(x.norm()*y.norm()+1e-8)

def collate(batch):
    ga,gb,go,dock,cell,syn,cls=zip(*batch)
    return (Batch.from_data_list(ga),Batch.from_data_list(gb),
            torch.stack(go),torch.stack(dock),
            torch.tensor(cell,dtype=torch.long),
            torch.tensor(syn,dtype=torch.float),
            torch.tensor(cls,dtype=torch.long))

class SynergyDataset(Dataset):
    def __init__(self,df,go_ctx,cell_line_to_idx):
        self.samples=[]; cache={}
        for _,row in tqdm(df.iterrows(),total=len(df),desc="Graphs"):
            for smi in [row['smiles_a'],row['smiles_b']]:
                if smi not in cache: cache[smi]=smiles_to_graph(smi)
            ga=cache.get(row['smiles_a']); gb=cache.get(row['smiles_b'])
            if ga is None or gb is None: continue
            go=go_ctx.get(row['uniprot'],torch.zeros(512))
            if go.dim()>1: go=go.squeeze(0)
            dock=torch.tensor([float(row['dock_score_a']),float(row['dock_score_b'])],dtype=torch.float)
            self.samples.append((ga,gb,go,dock,
                                 cell_line_to_idx.get(row['cell_line'],0),
                                 float(row['synergy_scaled']),int(row['synergy_class'])))
    def __len__(self): return len(self.samples)
    def __getitem__(self,i): return self.samples[i]

print("Building datasets...")
train_ds=SynergyDataset(train_df,go_ctx,cell_line_to_idx)
val_ds  =SynergyDataset(val_df,  go_ctx,cell_line_to_idx)
train_dl=DataLoader(train_ds,batch_size=128,shuffle=True, collate_fn=collate,num_workers=2)
val_dl  =DataLoader(val_ds,  batch_size=128,shuffle=False,collate_fn=collate,num_workers=2)

model    =ProteinSynergyDockV2(n_cell_lines=n_cell_lines).to(device)
optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=100)
mse=nn.MSELoss()
bce=nn.BCEWithLogitsLoss(pos_weight=torch.tensor([5.0]).to(device))

print(f"Params: {sum(p.numel() for p in model.parameters()):,}")
print("Training with CELL-LINE-HELD-OUT SPLIT — 100 epochs...\n")

best_r=best_auroc=-1.0
for epoch in range(100):
    model.train(); tloss=0
    for ga,gb,go,dock,cell,syn,cls in train_dl:
        ga,gb=ga.to(device),gb.to(device); go,dock=go.to(device),dock.to(device)
        cell=cell.to(device); syn,cls=syn.to(device),cls.to(device)
        optimizer.zero_grad()
        ps,pl=model(ga,gb,go,dock,cell)
        loss=mse(ps,syn)+0.5*bce(pl,cls.float())
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        optimizer.step(); tloss+=loss.item()
    scheduler.step(); tloss/=len(train_dl)

    model.eval(); all_ps,all_syn,all_pl,all_cls=[],[],[],[]
    with torch.no_grad():
        for ga,gb,go,dock,cell,syn,cls in val_dl:
            ga,gb=ga.to(device),gb.to(device); go,dock=go.to(device),dock.to(device)
            cell=cell.to(device); syn,cls=syn.to(device),cls.to(device)
            ps,pl=model(ga,gb,go,dock,cell)
            all_ps.append(ps.cpu()); all_syn.append(syn.cpu())
            all_pl.append(pl.cpu()); all_cls.append(cls.cpu())

    all_ps=torch.cat(all_ps); all_syn=torch.cat(all_syn)
    all_pl=torch.cat(all_pl); all_cls=torch.cat(all_cls)
    r=pearson_r(all_ps,all_syn).item()
    try: auroc=roc_auc_score(all_cls.numpy(),torch.sigmoid(all_pl).numpy())
    except: auroc=0.5

    if (epoch+1)%10==0 or epoch<5:
        print(f"Epoch {epoch+1:3d} | Loss: {tloss:.4f} | r: {r:.4f} | AUROC: {auroc:.4f}")

    if r>best_r:
        best_r=r; best_auroc=auroc
        torch.save({
            'epoch':epoch+1,'state_dict':model.state_dict(),
            'pearson_r':r,'auroc':auroc,
            'n_cell_lines':n_cell_lines,
            'cell_line_to_idx':cell_line_to_idx,
            'synergy_mean':float(scaler.mean_[0]),
            'synergy_std':float(scaler.scale_[0]),
        }, f'{WORK_DIR}/proteinsydock_v3_heldout.pt')
        if (epoch+1)%10==0 or epoch<5:
            print(f"  ✅ Best r={r:.4f} saved!")

print(f"\nFINAL (within training cell lines): r={best_r:.4f} | AUROC={best_auroc:.4f}")

# ═══════════════════════════════════════════════════════════════════════════
# THE ACTUAL RESULT: evaluate the best checkpoint on held-out cell lines
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("Evaluating on HELD-OUT cell lines (never seen during training)...")
print("="*70)

print("Building held-out test dataset...")
heldout_ds = SynergyDataset(heldout_df, go_ctx, cell_line_to_idx)
heldout_dl = DataLoader(heldout_ds, batch_size=128, shuffle=False, collate_fn=collate, num_workers=2)

ckpt = torch.load(f'{WORK_DIR}/proteinsydock_v3_heldout.pt', weights_only=False)
model.load_state_dict(ckpt['state_dict'])
model.eval()

ho_ps, ho_syn, ho_pl, ho_cls = [], [], [], []
with torch.no_grad():
    for ga, gb, go, dock, cell, syn, cls in heldout_dl:
        ga, gb = ga.to(device), gb.to(device)
        go, dock = go.to(device), dock.to(device)
        cell = cell.to(device)
        ps, pl = model(ga, gb, go, dock, cell)
        ho_ps.append(ps.cpu()); ho_syn.append(syn)
        ho_pl.append(pl.cpu()); ho_cls.append(cls)

ho_ps = torch.cat(ho_ps); ho_syn = torch.cat(ho_syn)
ho_pl = torch.cat(ho_pl); ho_cls = torch.cat(ho_cls)

ho_r = pearson_r(ho_ps, ho_syn).item()
try:
    ho_auroc = roc_auc_score(ho_cls.numpy(), torch.sigmoid(ho_pl).numpy())
except ValueError:
    ho_auroc = float('nan')

print("\n" + "="*70)
print("RESULT: Cell-Line-Held-Out Generalization Test")
print("="*70)
header_label = "Metric"
header_orig = "Random split (original)"
header_new = "Held-out split (this run)"
print(f"{header_label:<30}{header_orig:<28}{header_new}")
print(f"{'Pearson r':<30}{'0.5667':<28}{ho_r:.4f}")
print(f"{'AUROC':<30}{'0.7946':<28}{ho_auroc:.4f}")
print(f"{'N test samples':<30}{'~21,400 (random 20%)':<28}{len(heldout_df):,}")
print(f"{'N held-out cell lines':<30}{'0 (random split)':<28}{len(heldout_cell_lines)}")
print("="*70)
print(f"\nHeld-out cell lines tested: {sorted(heldout_cell_lines)}")
print(f"\nGeneralization gap (random_r - heldout_r) = {0.5667 - ho_r:.4f}")
print("A positive gap is expected — it means the random split was overstating")
print("true generalization to cancer types the model never trained on.")
print("This gap is itself the honest, citable finding from this experiment.")

results_summary = {
    "random_split": {"pearson_r": 0.5667, "auroc": 0.7946, "note": "original training run, epoch 82"},
    "heldout_split": {
        "pearson_r": float(ho_r),
        "auroc": float(ho_auroc),
        "n_test_samples": len(heldout_df),
        "heldout_cell_lines": sorted(heldout_cell_lines),
        "n_heldout_cell_lines": len(heldout_cell_lines),
        "n_train_samples": len(train_df),
        "n_val_samples": len(val_df),
    },
    "generalization_gap_pearson_r": float(0.5667 - ho_r),
}
with open(f'{WORK_DIR}/heldout_results.json', 'w') as f:
    json.dump(results_summary, f, indent=2)
print(f"\n✅ Saved {WORK_DIR}/heldout_results.json — download this file and share it back.")
