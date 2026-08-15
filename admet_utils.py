"""
admet_utils.py — ADMET property computation, chemical space, synergy models, and report generation.

Completely RDKit-based; no hard-coded property values. Every number computed from SMILES.
Importable without Streamlit.
"""
import math
import json
import numpy as np
from datetime import datetime
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, QED, AllChem, DataStructs
from rdkit.Chem import MolFromSmarts


# ── ADMET ─────────────────────────────────────────────────────────────────────

def compute_admet(smiles: str):
    """
    Compute a comprehensive ADMET profile from a SMILES string using RDKit.

    Returns a dict with physicochemical descriptors, Lipinski Rule of 5,
    ESOL estimated solubility (Delaney, 2004), BBB penetration estimate
    (Clark model), P-gp substrate heuristic, CYP3A4 substrate heuristic,
    QED drug-likeness, and a composite drug-likeness score.

    Returns None if SMILES is invalid.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    mw        = Descriptors.ExactMolWt(mol)
    logp      = Descriptors.MolLogP(mol)
    hbd       = rdMolDescriptors.CalcNumHBD(mol)
    hba       = rdMolDescriptors.CalcNumHBA(mol)
    tpsa      = Descriptors.TPSA(mol)
    rot_bonds = rdMolDescriptors.CalcNumRotatableBonds(mol)
    rings     = rdMolDescriptors.CalcNumRings(mol)
    arom_r    = rdMolDescriptors.CalcNumAromaticRings(mol)
    heavy     = mol.GetNumHeavyAtoms()
    fsp3      = rdMolDescriptors.CalcFractionCSP3(mol)
    qed_val   = QED.qed(mol)
    stereo    = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
    halogens  = len(mol.GetSubstructMatches(MolFromSmarts("[F,Cl,Br,I]")))

    # Lipinski Rule of 5 (allow 1 violation)
    lipinski = {
        "MW <= 500":   mw   <= 500,
        "LogP <= 5":   logp <= 5,
        "HBD <= 5":    hbd  <= 5,
        "HBA <= 10":   hba  <= 10,
    }
    violations = sum(1 for v in lipinski.values() if not v)
    lipinski_pass = violations <= 1

    # Veber bioavailability rules (rot_bonds <= 10, TPSA <= 140)
    veber_pass = rot_bonds <= 10 and tpsa <= 140

    # ESOL (Delaney 2004): log S = 0.16 - 0.63 cLogP - 0.0062 MW + 0.066 RB - 0.74 AP
    arom_prop  = arom_r / max(rings, 1) if rings > 0 else 0.0
    esol_log_s = 0.16 - 0.63 * logp - 0.0062 * mw + 0.066 * rot_bonds - 0.74 * arom_prop
    esol_mol_l = 10 ** esol_log_s
    if esol_mol_l < 1e-4:
        solubility_class = "Poor"
    elif esol_mol_l < 1e-2:
        solubility_class = "Moderate"
    else:
        solubility_class = "Good"

    # BBB Clark model heuristic (score 0-4; >=3 = penetrant)
    bbb_score    = int(tpsa < 90) + int(mw < 400) + int(0 <= logp <= 5) + int(hbd < 3)
    bbb_penetrant = bbb_score >= 3

    # P-glycoprotein substrate heuristic (Ecker 2001)
    pgp_substrate = mw > 400 and tpsa > 60

    # CYP3A4 substrate heuristic
    cyp3a4_substrate = mw > 300 and logp > 2 and arom_r >= 1

    # CYP2D6 inhibitor heuristic (basic N within aromatic system)
    cyp2d6_inhibitor = len(mol.GetSubstructMatches(MolFromSmarts("[#7;+0;!$(NC=O)]"))) > 0 and logp > 2

    # Composite drug-likeness (0-1, each criterion 1/7)
    dl_criteria = [
        mw <= 500, 0 <= logp <= 5, tpsa <= 140,
        rot_bonds <= 10, hbd <= 5, hba <= 10,
        violations <= 1
    ]
    dl_score = sum(dl_criteria) / len(dl_criteria)

    # Absorption score (simple: high logP + low TPSA = good passive permeability)
    absorption = max(0.0, min(1.0, (5 - tpsa / 140) * 0.5 + (logp / 5) * 0.5))

    return {
        # Physicochemical
        "MW":               round(mw, 2),
        "LogP":             round(logp, 2),
        "HBD":              hbd,
        "HBA":              hba,
        "TPSA":             round(tpsa, 2),
        "Rotatable Bonds":  rot_bonds,
        "Rings":            rings,
        "Aromatic Rings":   arom_r,
        "Heavy Atoms":      heavy,
        "Fsp3":             round(fsp3, 3),
        "Stereocenters":    stereo,
        "Halogens":         halogens,
        # Drug-likeness
        "QED":              round(qed_val, 3),
        "Drug-likeness":    round(dl_score, 2),
        "Lipinski":         lipinski,
        "Lipinski Pass":    lipinski_pass,
        "Lipinski Violations": violations,
        "Veber Pass":       veber_pass,
        # ADME
        "ESOL LogS":        round(esol_log_s, 2),
        "ESOL mol/L":       f"{esol_mol_l:.2e}",
        "Solubility":       solubility_class,
        "BBB Penetrant":    bbb_penetrant,
        "BBB Score":        bbb_score,
        "Pgp Substrate":    pgp_substrate,
        "CYP3A4 Substrate": cyp3a4_substrate,
        "CYP2D6 Inhibitor": cyp2d6_inhibitor,
        "Absorption":       round(absorption, 2),
    }


def get_pharmacophore_features(smiles: str) -> dict:
    """Extract pharmacophore-relevant features from SMILES using RDKit SMARTS."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {}
    return {
        "H-Bond Donors":      rdMolDescriptors.CalcNumHBD(mol),
        "H-Bond Acceptors":   rdMolDescriptors.CalcNumHBA(mol),
        "Aromatic Rings":     rdMolDescriptors.CalcNumAromaticRings(mol),
        "Hydrophobic Centers": len(mol.GetSubstructMatches(MolFromSmarts("[c,C;!$(C=O)]"))),
        "Basic Nitrogens":    len(mol.GetSubstructMatches(MolFromSmarts("[#7;+0;!$(NC=O);!$(nN);!$(Nn)]"))),
        "Acidic Groups":      len(mol.GetSubstructMatches(MolFromSmarts("[CX3](=O)[OH]"))),
        "Halogen Atoms":      len(mol.GetSubstructMatches(MolFromSmarts("[F,Cl,Br,I]"))),
        "Stereocenters":      len(Chem.FindMolChiralCenters(mol, includeUnassigned=True)),
        "Amide Bonds":        len(mol.GetSubstructMatches(MolFromSmarts("C(=O)N"))),
        "Sulfonamides":       len(mol.GetSubstructMatches(MolFromSmarts("S(=O)(=O)N"))),
    }


# ── Molecular Similarity ──────────────────────────────────────────────────────

def morgan_similarity(smiles_a: str, smiles_b: str, radius: int = 2, n_bits: int = 2048):
    """Tanimoto similarity via Morgan (ECFP4) fingerprints."""
    mol_a = Chem.MolFromSmiles(smiles_a)
    mol_b = Chem.MolFromSmiles(smiles_b)
    if mol_a is None or mol_b is None:
        return None
    fp_a = AllChem.GetMorganFingerprintAsBitVect(mol_a, radius, nBits=n_bits)
    fp_b = AllChem.GetMorganFingerprintAsBitVect(mol_b, radius, nBits=n_bits)
    return round(DataStructs.TanimotoSimilarity(fp_a, fp_b), 4)


def compute_all_fingerprints(smiles_dict: dict, radius: int = 2, n_bits: int = 1024) -> dict:
    """Morgan fingerprints for every valid SMILES in a name->smiles dict."""
    fps = {}
    for name, smiles in smiles_dict.items():
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
            fps[name] = np.array(fp, dtype=np.float32)
    return fps


def chemical_space_pca(smiles_dict: dict, highlight=None):
    """
    PCA of Morgan fingerprints for all drugs in smiles_dict.

    Returns (coords_dict, variance_explained) where coords_dict maps
    drug name -> {"x": float, "y": float, "highlighted": bool}.
    Requires scikit-learn.
    """
    try:
        from sklearn.decomposition import PCA
    except ImportError:
        return {}, (0.0, 0.0)

    fps = compute_all_fingerprints(smiles_dict)
    if len(fps) < 3:
        return {}, (0.0, 0.0)

    names = list(fps.keys())
    X = np.array([fps[n] for n in names])
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X)
    var_exp = tuple(float(v) for v in pca.explained_variance_ratio_)

    result = {
        names[i]: {
            "x": float(coords[i, 0]),
            "y": float(coords[i, 1]),
            "highlighted": bool(highlight and names[i] in highlight),
        }
        for i in range(len(names))
    }
    return result, var_exp


def tanimoto_matrix(smiles_dict: dict):
    """
    Compute pairwise Tanimoto similarity matrix for all drugs.

    Returns (names, matrix_np_array).
    """
    names = list(smiles_dict.keys())
    n = len(names)
    mat = np.zeros((n, n), dtype=np.float32)
    rdkit_fps = {}
    for name in names:
        mol = Chem.MolFromSmiles(smiles_dict[name])
        if mol:
            rdkit_fps[name] = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 1024)

    for i, ni in enumerate(names):
        for j, nj in enumerate(names):
            if ni in rdkit_fps and nj in rdkit_fps:
                mat[i, j] = DataStructs.TanimotoSimilarity(rdkit_fps[ni], rdkit_fps[nj])
    return names, mat


def similarity_interpretation(tanimoto: float):
    """Human-readable interpretation of Tanimoto similarity for drug pairs."""
    if tanimoto >= 0.85:
        return "🔴 Near-identical", "Very high structural similarity — drugs likely share binding mode, high competition/antagonism risk."
    elif tanimoto >= 0.65:
        return "🟠 Highly similar", "Substantial structural overlap — possible binding competition; verify mechanistically."
    elif tanimoto >= 0.40:
        return "🟡 Moderately similar", "Some shared scaffolds — could be complementary or overlapping depending on target."
    elif tanimoto >= 0.20:
        return "🟢 Diverse", "Structurally distinct — good starting point for synergy exploration."
    else:
        return "🟢 Highly diverse", "Very different chemical series — complementary mechanisms likely."


# ── Synergy Models ────────────────────────────────────────────────────────────

def bliss_analysis(score_a: float, score_b: float, combined_score: float) -> dict:
    """
    Bliss Independence synergy model.

    Converts synergy scores to effect probabilities via sigmoid,
    then computes Bliss-expected combination effect and deviation.

    Reference: Bliss CI (1939). The toxicity of poisons applied jointly.
    """
    def to_prob(s):
        return 1.0 / (1.0 + math.exp(-s))

    ea   = to_prob(score_a)
    eb   = to_prob(score_b)
    e_ab = to_prob(combined_score)
    bliss_expected = ea + eb - ea * eb
    deviation = e_ab - bliss_expected

    if deviation > 0.05:
        classification = "🟢 Synergistic (Bliss)"
        color = "green"
    elif deviation < -0.05:
        classification = "🔴 Antagonistic (Bliss)"
        color = "red"
    else:
        classification = "🟡 Additive (Bliss)"
        color = "orange"

    return {
        "E_A":            round(ea, 4),
        "E_B":            round(eb, 4),
        "E_AB_expected":  round(bliss_expected, 4),
        "E_AB_observed":  round(e_ab, 4),
        "Bliss_deviation": round(deviation, 4),
        "Classification": classification,
        "color":          color,
    }


def combination_index(synergy_score: float):
    """
    Approximate Chou-Talalay Combination Index from a GNN synergy score.

    Mapping: CI = exp(-synergy_score).
    CI < 1 = synergy, CI = 1 = additive, CI > 1 = antagonism.

    Reference: Chou TC (2010). Drug combination studies and their synergy
    quantification using the Chou-Talalay method. Cancer Res 70(2):440-6.
    """
    ci = math.exp(-synergy_score)
    if ci < 0.3:
        label, color = "Strong Synergy",     "green"
    elif ci < 0.7:
        label, color = "Synergy",            "green"
    elif ci < 0.9:
        label, color = "Slight Synergy",     "lightgreen"
    elif ci < 1.1:
        label, color = "Additive",           "gray"
    elif ci < 1.45:
        label, color = "Slight Antagonism",  "orange"
    elif ci < 3.3:
        label, color = "Antagonism",         "red"
    else:
        label, color = "Strong Antagonism",  "darkred"
    return round(ci, 3), label, color


def dose_response_hill(ec50: float, hill: float = 1.5, n: int = 120, max_c: float = 100.0):
    """
    Hill equation dose-response curve.
    Effect(C) = C^n / (EC50^n + C^n) x 100%
    Returns (concentrations_list, responses_list).
    """
    concs = np.logspace(-3, math.log10(max_c), n)
    resps = concs**hill / (ec50**hill + concs**hill) * 100.0
    return concs.tolist(), resps.tolist()


def synergy_dose_matrix(ec50_a: float, ec50_b: float, synergy_score: float, n_doses: int = 8):
    """
    Generate a dose-effect matrix for two drugs, shifted by synergy score.

    Returns (doses_a, doses_b, matrix) where matrix[i][j] is the predicted
    combined effect at dose_a[i], dose_b[j] relative to Bliss independence.
    """
    def hill(c, ec50, h=1.5):
        return c**h / (ec50**h + c**h)

    doses_a = list(np.logspace(-2, 2, n_doses))
    doses_b = list(np.logspace(-2, 2, n_doses))
    synergy_boost = math.tanh(synergy_score) * 0.25  # bounded +/-0.25

    matrix = []
    for ca in doses_a:
        row = []
        for cb in doses_b:
            ea = hill(ca, ec50_a)
            eb = hill(cb, ec50_b)
            bliss = ea + eb - ea * eb
            combined = min(1.0, max(0.0, bliss + synergy_boost))
            row.append(round(combined * 100, 1))
        matrix.append(row)

    return [round(d, 3) for d in doses_a], [round(d, 3) for d in doses_b], matrix


# ── Model Explainability ──────────────────────────────────────────────────────

def atom_feature_importance(smiles: str):
    """
    Estimate per-atom contribution to drug binding using RDKit atomic properties.

    Returns a list of dicts {atom_idx, symbol, importance, rationale}
    ranked by estimated importance. Uses interpretable chemistry rules
    (not backprop) so it works without requiring gradients.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    results = []
    for atom in mol.GetAtoms():
        idx      = atom.GetIdx()
        symbol   = atom.GetSymbol()
        aromatic = atom.GetIsAromatic()
        in_ring  = atom.IsInRing()
        formal_q = atom.GetFormalCharge()
        hbd      = 1 if (symbol in ('N', 'O') and atom.GetTotalNumHs() > 0) else 0
        hba      = 1 if symbol in ('N', 'O', 'F', 'S') else 0
        degree   = atom.GetDegree()

        importance = 0.0
        rationale_parts = []

        if aromatic and symbol != 'C':
            importance += 0.35
            rationale_parts.append("aromatic heteroatom")
        elif aromatic:
            importance += 0.15
            rationale_parts.append("aromatic carbon")
        if hbd:
            importance += 0.25
            rationale_parts.append("H-bond donor")
        if hba and not hbd:
            importance += 0.15
            rationale_parts.append("H-bond acceptor")
        if formal_q != 0:
            importance += 0.20
            rationale_parts.append("charged atom")
        if in_ring and symbol not in ('C',):
            importance += 0.10
            rationale_parts.append("ring heteroatom")
        if degree >= 3 and not aromatic:
            importance += 0.05
            rationale_parts.append("branch point")

        results.append({
            "atom_idx":   idx,
            "symbol":     symbol,
            "importance": round(importance, 3),
            "rationale":  ", ".join(rationale_parts) if rationale_parts else "aliphatic carbon",
        })

    return sorted(results, key=lambda x: x["importance"], reverse=True)


def drug_feature_importance_bar(smiles: str, drug_name: str) -> dict:
    """
    Aggregate atom importances into feature-class importance scores.
    Returns a dict suitable for a Plotly bar chart.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {}

    features = {
        "Aromatic heteroatoms": 0.0,
        "Aromatic carbons":     0.0,
        "H-bond donors":        0.0,
        "H-bond acceptors":     0.0,
        "Charged atoms":        0.0,
        "Ring systems":         0.0,
        "Aliphatic chains":     0.0,
    }
    atoms = atom_feature_importance(smiles) or []
    for a in atoms:
        r = a["rationale"]
        imp = a["importance"]
        if "aromatic heteroatom" in r:
            features["Aromatic heteroatoms"] += imp
        elif "aromatic carbon" in r:
            features["Aromatic carbons"] += imp
        if "H-bond donor" in r:
            features["H-bond donors"] += imp
        if "H-bond acceptor" in r:
            features["H-bond acceptors"] += imp
        if "charged" in r:
            features["Charged atoms"] += imp
        if "ring heteroatom" in r:
            features["Ring systems"] += imp
        if imp == 0.0:
            features["Aliphatic chains"] += 0.05

    total = sum(features.values()) or 1
    return {k: round(v / total, 3) for k, v in features.items()}


# ── HTML Report Generator ─────────────────────────────────────────────────────

def generate_html_report(d: dict) -> str:
    """
    Generate a standalone, dark-themed HTML report from a prediction data dict.
    """
    ts      = d.get("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M UTC"))
    da      = d.get("drug_a", "Drug A")
    db      = d.get("drug_b", "Drug B")
    syn     = d.get("synergy_score", 0.0)
    std     = d.get("synergy_std", 0.0)
    prob    = d.get("synergy_prob", 0.0)
    dock_a  = d.get("dock_a", 0.0)
    dock_b  = d.get("dock_b", 0.0)
    verdict = d.get("verdict", "Unknown")
    ci      = d.get("ci", "N/A")
    ci_l    = d.get("ci_label", "N/A")
    tan     = d.get("tanimoto", None)
    tan_str = f"{tan:.3f}" if tan is not None else "N/A"

    admet_props = ["MW","LogP","HBD","HBA","TPSA","QED",
                   "Solubility","BBB Penetrant","Lipinski Pass","Drug-likeness"]
    admet_a = d.get("admet_a", {})
    admet_b = d.get("admet_b", {})

    admet_rows = ""
    for prop in admet_props:
        va = admet_a.get(prop, "N/A")
        vb = admet_b.get(prop, "N/A")
        admet_rows += f"<tr><td>{prop}</td><td>{va}</td><td>{vb}</td></tr>\n"

    bliss = d.get("bliss", {})
    bliss_html = ""
    if bliss:
        bliss_html = f"""
        <h2>Bliss Independence Model</h2>
        <div class="card">
        <table>
        <tr><th>Property</th><th>Value</th></tr>
        <tr><td>E(Drug A alone)</td><td>{bliss.get('E_A','N/A')}</td></tr>
        <tr><td>E(Drug B alone)</td><td>{bliss.get('E_B','N/A')}</td></tr>
        <tr><td>Expected combined (Bliss)</td><td>{bliss.get('E_AB_expected','N/A')}</td></tr>
        <tr><td>Observed combined</td><td>{bliss.get('E_AB_observed','N/A')}</td></tr>
        <tr><td>Bliss deviation</td><td>{bliss.get('Bliss_deviation','N/A')}</td></tr>
        <tr><td>Classification</td><td><b>{bliss.get('Classification','N/A')}</b></td></tr>
        </table>
        </div>"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ProteinSynergyDock Report -- {da} + {db}</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0;}}
body{{font-family:'Segoe UI',Arial,sans-serif;background:#0d1117;color:#c9d1d9;padding:24px;line-height:1.6;}}
h1{{color:#4fc3f7;font-size:1.9rem;margin-bottom:4px;}}
h2{{color:#81d4fa;font-size:1.25rem;border-bottom:1px solid #30363d;padding-bottom:6px;margin:24px 0 12px;}}
.subtitle{{color:#8b949e;font-size:0.9rem;margin-bottom:20px;}}
.card{{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:16px;margin:10px 0;}}
.metrics{{display:flex;flex-wrap:wrap;gap:12px;margin:12px 0;}}
.metric{{background:#1c2128;border:1px solid #30363d;border-radius:8px;padding:14px 22px;text-align:center;min-width:140px;}}
.metric .val{{font-size:1.5rem;font-weight:700;color:#4fc3f7;display:block;}}
.metric .lab{{font-size:0.78rem;color:#8b949e;margin-top:4px;display:block;}}
table{{width:100%;border-collapse:collapse;font-size:0.9rem;}}
td,th{{padding:8px 12px;border-bottom:1px solid #21262d;}}
th{{color:#8b949e;font-weight:600;text-align:left;background:#161b22;}}
tr:hover td{{background:#1c2128;}}
.warn{{background:#2d1f00;border:1px solid #f0883e;color:#f0883e;border-radius:6px;padding:10px 14px;margin:8px 0;font-size:0.9rem;}}
footer{{text-align:center;color:#484f58;margin-top:40px;font-size:0.82rem;border-top:1px solid #21262d;padding-top:16px;}}
@media print{{body{{background:white;color:black;}}.card{{border:1px solid #ccc;}}h1,h2{{color:#333;}}}}
</style>
</head>
<body>
<h1>🧬 ProteinSynergyDock Report</h1>
<p class="subtitle">Generated: {ts} &nbsp;|&nbsp; Research tool only -- not for clinical use</p>
<h2>Drug Combination</h2>
<div class="card">
  <div class="metrics">
    <div class="metric"><span class="val">{da}</span><span class="lab">Drug A</span></div>
    <div class="metric"><span class="val">{db}</span><span class="lab">Drug B</span></div>
    <div class="metric"><span class="val">{d.get('pdb_id','N/A')}</span><span class="lab">PDB Target</span></div>
    <div class="metric"><span class="val">{d.get('cell_line','N/A')}</span><span class="lab">Cell Line</span></div>
    <div class="metric"><span class="val">{d.get('panel','N/A')}</span><span class="lab">Cancer Type</span></div>
    <div class="metric"><span class="val">{tan_str}</span><span class="lab">Tanimoto Similarity</span></div>
  </div>
</div>
<h2>Synergy Prediction</h2>
<div class="card">
  <div class="metrics">
    <div class="metric"><span class="val">{syn:.3f} +/- {std:.3f}</span><span class="lab">Synergy Score (MC Dropout)</span></div>
    <div class="metric"><span class="val">{prob:.3f}</span><span class="lab">Synergy Probability</span></div>
    <div class="metric"><span class="val">{dock_a:.2f} kcal/mol</span><span class="lab">{da} Binding</span></div>
    <div class="metric"><span class="val">{dock_b:.2f} kcal/mol</span><span class="lab">{db} Binding</span></div>
    <div class="metric"><span class="val">{ci}</span><span class="lab">Combination Index (CI)</span></div>
    <div class="metric"><span class="val">{ci_l}</span><span class="lab">CI Classification</span></div>
  </div>
  <p style="margin-top:10px"><b>Verdict:</b> {verdict}</p>
</div>
<h2>Model Details</h2>
<div class="card">
<table>
<tr><th>Property</th><th>Value</th></tr>
<tr><td>Protein</td><td>{d.get('protein_name','N/A')}</td></tr>
<tr><td>Model Version</td><td>{str(d.get('model_version','N/A')).upper()}</td></tr>
<tr><td>Pearson r (held-out)</td><td>{d.get('model_r',0):.4f}</td></tr>
<tr><td>AUROC (held-out)</td><td>{d.get('model_auroc',0):.4f}</td></tr>
<tr><td>MC Dropout Samples</td><td>{d.get('mc_samples',20)}</td></tr>
<tr><td>Training data</td><td>107,103 NCI ALMANAC triplets</td></tr>
</table>
</div>
<h2>ADMET Properties</h2>
<div class="card">
<table>
<tr><th>Property</th><th>{da}</th><th>{db}</th></tr>
{admet_rows}
</table>
</div>
{bliss_html}
<h2>Interpretation Guide</h2>
<div class="card">
<table>
<tr><th>Synergy Score</th><th>Meaning</th></tr>
<tr><td>&gt; 0.5</td><td>Strongly Synergistic</td></tr>
<tr><td>0.1 to 0.5</td><td>Mildly Synergistic</td></tr>
<tr><td>-0.1 to 0.1</td><td>Approximately Additive</td></tr>
<tr><td>&lt; -0.1</td><td>Antagonistic</td></tr>
</table>
<br>
<table>
<tr><th>CI</th><th>Chou-Talalay Classification</th></tr>
<tr><td>&lt; 0.3</td><td>Strong Synergy</td></tr>
<tr><td>0.3 to 0.7</td><td>Synergy</td></tr>
<tr><td>0.7 to 0.9</td><td>Slight Synergy</td></tr>
<tr><td>0.9 to 1.1</td><td>Additive</td></tr>
<tr><td>1.1 to 1.45</td><td>Slight Antagonism</td></tr>
<tr><td>1.45 to 3.3</td><td>Antagonism</td></tr>
<tr><td>&gt; 3.3</td><td>Strong Antagonism</td></tr>
</table>
</div>
<div class="warn">Warning: This is a research tool, not a clinical diagnostic. Not FDA-reviewed. Consult qualified medical professionals for clinical decisions.</div>
<footer>ProteinSynergyDock &nbsp;|&nbsp; Aprameya Bharadwaj &nbsp;|&nbsp;
<a href="https://github.com/Aprameya05/ProteinSynergyDock-App" style="color:#4fc3f7">
github.com/Aprameya05/ProteinSynergyDock-App</a></footer>
</body>
</html>"""
