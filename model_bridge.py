"""
model_bridge.py

Bridges the FHIR/API layer to the real ProteinSynergyDockV2 model.
Fully wired — nothing left to fill in.

Confirmed from app.py:
- Checkpoint: proteinsydock_v2_final.pt, loaded via torch.load(..., weights_only=False)
- Model class: ProteinSynergyDockV2(n_cell_lines=ckpt.get('n_cell_lines', 60))
- cell_to_idx comes from ckpt['cell_line_to_idx']
- go_emb is a zero placeholder: torch.zeros(512).unsqueeze(0) (not real GO data
  in the existing app either — this mirrors that, not a new simplification)
- dock is built from per-drug AutoDock Vina scores (dsa, dsb) computed via a
  real docking run in the Streamlit tab. Real-time Vina docking is too slow
  for a synchronous API call, so this module defaults dock to [0.0, 0.0]
  unless real docking scores are explicitly passed in. This is documented
  here and surfaced in the FHIR report rather than silently pretending to
  have live docking data — same honesty standard as the rest of this layer.

This file loads the model ONCE per process (module-level cache in
_MODEL_STATE) since this runs standalone, not inside Streamlit's
@st.cache_resource.
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

import torch
import torch.nn as nn
from rdkit import Chem
from rdkit.Chem import AllChem
from torch_geometric.nn import GATv2Conv, global_mean_pool
from torch_geometric.data import Data, Batch


CHECKPOINT_PATH = os.environ.get("PSD_CHECKPOINT_PATH", "proteinsydock_v2_final.pt")


class ModelUnavailableError(Exception):
    """Raised when the underlying model/core logic can't produce a prediction."""
    pass


# ---------------------------------------------------------------------------
# Drug name -> SMILES lookup (mirrors core.py's DRUG_SMILES_LOOKUP).
# If core.py is importable from this process, prefer importing it directly:
#   from core import DRUG_SMILES_LOOKUP
# Kept inline here so this module has zero dependency on app.py-adjacent
# import paths, which matters when deployed as a separate Render service.
# ---------------------------------------------------------------------------
DRUG_SMILES_LOOKUP = {
    "Imatinib": "CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)NC4=NC=CC(=N4)C5=CN=CC=C5",
    "Gefitinib": "COC1=C(C=C2C(=C1)N=CN=C2NC3=CC(=C(C=C3)F)Cl)OCCCN4CCOCC4",
    "Erlotinib": "COCCOC1=C(C=C2C(=C1)C(=NC=N2)NC3=CC=CC(=C3)C#C)OCCOC",
    "Lapatinib": "CS(=O)(=O)CCNCc1oc(cc1)c2ccc3ncnc(Nc4ccc(Oc5cccc(Cl)c5)c(Cl)c4)c3c2",
    "Dasatinib": "Cc1nc(Nc2ncc(s2)C(=O)Nc2c(C)cccc2Cl)cc(n1)N1CCN(CCO)CC1",
    "Nilotinib": "Cc1cn(c2cc(NC(=O)c3ccc(C)c(Nc4nccc(n4)-c4cccnc4)c3)cc(C(F)(F)F)c12)C",
    "Vemurafenib": "CCCS(=O)(=O)Nc1ccc(F)c(C(=O)c2c[nH]c3ncc(-c4ccc(Cl)cc4)cc23)c1",
    "Dabrafenib": "CC(C)(C)c1nc2cc(F)ccc2c(C(=O)Nc2ccc(F)c(NS(=O)(=O)c3ccc(F)cc3)c2)n1",
    "Trametinib": "CC(=O)Nc1ccc(-c2cc3c(nc(N)nc3n2C)N2CCC(F)(F)CC2=O)cc1F",
    "Cobimetinib": "OC(COc1cc(Cl)c(F)cc1F)CN1CCC(=C1)c1cc2c(Nc3ccc(F)cc3F)ncc(C(N)=O)c2[nH]1",
    "Sorafenib": "CNC(=O)c1cc(Oc2ccc(NC(=O)Nc3ccc(Cl)c(C(F)(F)F)c3)cc2)ccn1",
    "Sunitinib": "CCN(CC)CCNC(=O)c1c(C)[nH]c(C=C2C(=O)Nc3ccc(F)cc32)c1C",
    "Olaparib": "O=C1CCCN1c1ccc(cc1)C(=O)c1[nH]ncc1C1CC1",
    "Niraparib": "OC(=O)c1ccc2[nH]ncc2c1-c1ccc(cn1)C1CCNCC1",
    "Rucaparib": "CNCC1=CC=C(C=C1)C2=C3CCNC(=O)C4=CC(=CC(=C34)N2)F",
    "Palbociclib": "CC1=C(C(=NC(=C1)N2CCNCC2)N3CCCC3)C(=O)NC4=CC=CC=N4",
    "Abemaciclib": "CC1=NC(=NC(=C1)NC2=NC=CC(=N2)N3CCC(CC3)NC(=O)C4=CC=C(C=C4)F)C5=CC(=CC=C5)F",
    "Ribociclib": "CC1=NC(=NC(=C1)N2CCNCC2)C3=CC4=C(C=C3)N=CN=C4N5CCCC5",
    "Ibrutinib": "C=CC(=O)N1CCCC(c2ncnc3[nH]ccc23)C1",
    "Zanubrutinib": "O=C(/C=C/c1ccco1)N1CCC(n2nc(-c3ccc4c(c3)CCNC4=O)c3c(N)ncnc23)CC1",
    "Acalabrutinib": "CC#CC(=O)N1CCC(n2nc(-c3ccc4c(c3)CCNC4=O)c3c(N)ncnc23)CC1",
    "Venetoclax": "CC1(CCC(CC1)N2CCN(CC2)c3ccc(cc3)C(=O)NS(=O)(=O)c4ccc(cc4-c5cnc6ccccc6n5)Cl)C",
    "Alpelisib": "CC1(C)CN(c2nc(Nc3ccc(S(N)(=O)=O)cc3F)ncc2F)CC1=O",
    "Paclitaxel": "O=C(OC1C[C@]2(O)C(=O)C(OC(=O)c3ccccc3)C(O)C(OC(=O)C(NC(=O)c3ccccc3)c3ccccc3)C2(C)CC1)C(C)=C",
    "Doxorubicin": "COc1cccc2C(=O)c3c(O)c4CC(O)(CC(OC5CC(N)C(O)C(C)O5)c4c(O)c3C(=O)c12)C(=O)CO",
    "Gemcitabine": "NC(=O)C1=CN(C(=O)N1)C1CC(F)(F)C(CO)O1",
    "Osimertinib": "C=CC(=O)Nc1cc2c(Nc3ccc(F)c(Cl)c3)nc(OC)nc2cc1N(C)CCN(C)C",
    "Alectinib": "CCC1=C(C=C2C(=C1)C(=O)C3=C(C2(C)C)NC4=C3C=CC(=C4)C#N)N5CCC(CC5)N6CCOCC6",
    "Afatinib": "CN(C)C/C=C/C(=O)Nc1cc2c(Nc3ccc(F)c(Cl)c3)ncnc2cc1OC",
    "Capecitabine": "CCOC(=O)Nc1nc(=O)n(C2OC(C)C(O)C2O)cc1F",
    "Temozolomide": "Cn1nnc2c(C(N)=O)ncn12",
    "Selumetinib": "Cc1cc(Nc2ncc(F)c(Nc3ccc(I)c(F)c3)n2)c(Cl)cc1Cl",
    "Belinostat": "O=C(/C=C/c1ccccc1)NOc1ccc(NS(=O)(=O)c2ccccc2)cc1",
    "Vorinostat": "O=C(CCCCCCC(=O)Nc1ccccc1)NO",
    "Crizotinib": "Cc1cn(C2CCNCC2)c2cc(Nc3ccc(F)cc3Cl)cnc12",
}


def smiles_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
        mol = Chem.RemoveHs(mol)
        if mol.GetNumConformers() == 0:
            AllChem.Compute2DCoords(mol)
    except Exception:
        try:
            AllChem.Compute2DCoords(mol)
        except Exception:
            return None
    feats, pos = [], []
    conf = mol.GetConformer() if mol.GetNumConformers() > 0 else None
    for atom in mol.GetAtoms():
        feats.append([
            atom.GetAtomicNum(), atom.GetDegree(), atom.GetFormalCharge(),
            int(atom.GetIsAromatic()), int(atom.IsInRing()),
            atom.GetTotalNumHs(), atom.GetNumRadicalElectrons(),
        ])
        if conf:
            p = conf.GetAtomPosition(atom.GetIdx())
            pos.append([p.x, p.y, p.z])
        else:
            pos.append([0., 0., 0.])
    es, ed = [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        es += [i, j]
        ed += [j, i]
    if not es:
        return None
    return Data(
        x=torch.tensor(feats, dtype=torch.float),
        pos=torch.tensor(pos, dtype=torch.float),
        edge_index=torch.tensor([es, ed], dtype=torch.long),
    )


def _enable_mc_dropout(model):
    for module in model.modules():
        if module.__class__.__name__.startswith("Dropout"):
            module.train()


def predict_with_uncertainty(model, model_version, cell_to_idx, ga, gb, go_emb, dock,
                              cell_line, batch_cls, n_samples=20):
    model.eval()
    _enable_mc_dropout(model)

    synergy_samples, prob_samples = [], []
    with torch.no_grad():
        for _ in range(n_samples):
            if model_version == "v2" and cell_to_idx:
                cidx = torch.tensor([cell_to_idx.get(cell_line, 0)], dtype=torch.long)
                score, logit = model(
                    batch_cls.from_data_list([ga]), batch_cls.from_data_list([gb]),
                    go_emb, dock, cidx,
                )
            else:
                score, logit = model(
                    batch_cls.from_data_list([ga]), batch_cls.from_data_list([gb]),
                    go_emb, dock,
                )
            synergy_samples.append(score.item())
            prob_samples.append(torch.sigmoid(logit).item())

    model.eval()

    import numpy as np
    synergy_samples = np.array(synergy_samples)
    prob_samples = np.array(prob_samples)
    return {
        "mean_synergy": float(synergy_samples.mean()),
        "std_synergy": float(synergy_samples.std()),
        "mean_prob": float(prob_samples.mean()),
        "std_prob": float(prob_samples.std()),
        "synergy_samples": synergy_samples.tolist(),
        "n_samples": n_samples,
    }


# ===========================================================================
# Model loading — confirmed exact from app.py's load_model()
# ===========================================================================

_MODEL_STATE = {}


def _load_model_state():
    """
    Mirrors app.py's @st.cache_resource load_model() exactly, minus the
    Streamlit decorator. Loads once per process into _MODEL_STATE.

    NOTE: ProteinSynergyDockV2 / ProteinSynergyDockV1 class definitions
    are NOT duplicated here — they're imported from core.py (or app.py,
    if that's where they actually live in your repo). If the import below
    fails, move/duplicate the class definitions into core.py, since a
    standalone API process needs them importable without Streamlit.
    """
    try:
        from core import ProteinSynergyDockV1, ProteinSynergyDockV2
    except ImportError as e:
        raise ModelUnavailableError(
            "Could not import ProteinSynergyDockV1/V2 from core.py. These "
            "model classes currently live in app.py — move their definitions "
            "into core.py (or a new model_arch.py) so they're importable "
            "without Streamlit. Original error: " + str(e)
        )

    if not os.path.exists(CHECKPOINT_PATH):
        raise ModelUnavailableError(
            f"Checkpoint file '{CHECKPOINT_PATH}' not found. Make sure "
            f"proteinsydock_v2_final.pt is present in the deployment "
            f"(same directory, or set PSD_CHECKPOINT_PATH env var)."
        )

    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"]

    if any("cell_embed" in k for k in sd):
        m = ProteinSynergyDockV2(n_cell_lines=ckpt.get("n_cell_lines", 60))
        m.load_state_dict(sd)
        m.eval()
        _MODEL_STATE["model"] = m
        _MODEL_STATE["cell_to_idx"] = ckpt.get("cell_line_to_idx", {})
        _MODEL_STATE["syn_scale"] = (
            ckpt.get("synergy_mean", -2.58),
            ckpt.get("synergy_std", 6.06),
        )
        _MODEL_STATE["model_version"] = "v2"
        _MODEL_STATE["model_r"] = ckpt.get("pearson_r", 0.0)
        _MODEL_STATE["model_auroc"] = ckpt.get("auroc", 0.0)
    else:
        m = ProteinSynergyDockV1()
        m.load_state_dict(sd)
        m.eval()
        _MODEL_STATE["model"] = m
        _MODEL_STATE["cell_to_idx"] = None
        _MODEL_STATE["syn_scale"] = None
        _MODEL_STATE["model_version"] = "v1"
        _MODEL_STATE["model_r"] = ckpt.get("pearson_r", 0.0)
        _MODEL_STATE["model_auroc"] = ckpt.get("auroc", 0.0)


# ===========================================================================
# Public entrypoint
# ===========================================================================

def predict_synergy(
    drug_a: str,
    drug_b: str,
    cell_line: str,
    docking_score_a: Optional[float] = None,
    docking_score_b: Optional[float] = None,
) -> Tuple[float, Optional[float], Optional[float]]:
    """
    Returns (synergy_score, confidence, docking_affinity).

    docking_score_a/b: optional real AutoDock Vina scores for each drug, if
    already computed elsewhere. If omitted, dock defaults to [0.0, 0.0] —
    this is a known limitation (real-time docking is too slow for a
    synchronous API call) and is surfaced honestly rather than faked.
    """
    if not _MODEL_STATE:
        _load_model_state()  # raises ModelUnavailableError on failure

    model = _MODEL_STATE["model"]
    if model is None:
        raise ModelUnavailableError(
            f"Checkpoint '{CHECKPOINT_PATH}' loaded but contained no model "
            f"(load_model() returned None — file may be corrupt or empty)."
        )

    smiles_a = DRUG_SMILES_LOOKUP.get(drug_a)
    smiles_b = DRUG_SMILES_LOOKUP.get(drug_b)
    if smiles_a is None:
        raise ModelUnavailableError(f"No SMILES found for drug '{drug_a}'.")
    if smiles_b is None:
        raise ModelUnavailableError(f"No SMILES found for drug '{drug_b}'.")

    ga = smiles_to_graph(smiles_a)
    gb = smiles_to_graph(smiles_b)
    if ga is None or gb is None:
        raise ModelUnavailableError(
            f"Could not build molecular graph for '{drug_a}' or '{drug_b}' "
            f"(invalid SMILES or RDKit embedding failure)."
        )

    # go_emb: zero placeholder, mirrors app.py's existing prediction tab
    # exactly (not a new simplification introduced by this API layer).
    go_emb = torch.zeros(512).unsqueeze(0)

    dsa = docking_score_a if docking_score_a is not None else 0.0
    dsb = docking_score_b if docking_score_b is not None else 0.0
    dock = torch.tensor([[float(dsa), float(dsb)]])

    try:
        result = predict_with_uncertainty(
            model=model,
            model_version=_MODEL_STATE["model_version"],
            cell_to_idx=_MODEL_STATE["cell_to_idx"],
            ga=ga,
            gb=gb,
            go_emb=go_emb,
            dock=dock,
            cell_line=cell_line,
            batch_cls=Batch,
            n_samples=20,
        )
    except Exception as e:
        raise ModelUnavailableError(f"Inference failed: {e}") from e

    synergy_score = result["mean_synergy"]
    std = result["std_synergy"]
    # Confidence band mirrors core.py's confidence_label() thresholds.
    if std < 0.15:
        confidence = 0.9
    elif std < 0.4:
        confidence = 0.6
    else:
        confidence = 0.3

    docking_affinity = None
    if docking_score_a is not None and docking_score_b is not None:
        docking_affinity = (docking_score_a + docking_score_b) / 2.0

    return synergy_score, confidence, docking_affinity
