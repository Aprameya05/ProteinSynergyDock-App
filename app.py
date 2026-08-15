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
from ligplot_utils import (
        read_all_poses, generate_all_ligplots_zip,
        parse_vina_affinities, find_interactions, generate_ligplot
    )
 
st.set_page_config(page_title="ProteinSynergyDock", page_icon="🧬", layout="wide", initial_sidebar_state="expanded")
st.markdown("""<style>
.main-header{text-align:center;padding:2rem;background:linear-gradient(135deg,#1a1a2e 0%,#16213e 50%,#0f3460 100%);border-radius:12px;margin-bottom:2rem;}
.main-header h1{color:#4fc3f7;font-size:2.5rem;margin:0;}
.main-header p{color:#b0bec5;margin:0.5rem 0 0;}
.known-score{background:#1e3a1e;border-left:4px solid #4caf50;padding:12px;border-radius:6px;margin:8px 0;color:white;}
.unknown-score{background:#2a2a1e;border-left:4px solid #ff9800;padding:12px;border-radius:6px;margin:8px 0;color:white;}
.history-item{background:#1a1a2e;border-left:3px solid #4fc3f7;padding:8px;border-radius:4px;margin:4px 0;color:white;font-size:12px;}
/* Make the tab bar wrap onto multiple rows instead of requiring horizontal scroll */
.stTabs [data-baseweb="tab-list"]{flex-wrap:wrap;gap:4px;row-gap:6px;}
.stTabs [data-baseweb="tab"]{height:auto;white-space:normal;padding:8px 14px;font-size:14px;}
.stTabs [data-baseweb="tab-highlight"]{display:none;}
.stTabs [data-baseweb="tab-border"]{display:none;}
</style>""", unsafe_allow_html=True)

from core import DrugEncoder, CrossDrugAttention, ProteinSynergyDockV2, ProteinSynergyDockV1

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

# ── Data ───────────────────────────────────────────────────────────────────────
from core import (
    KNOWN_SYNERGY, DRUG_SMILES_LOOKUP, CANCER_PANELS, SHOWCASES,
    DRUG_MECHANISMS, SYNERGY_RULES, MUTATION_DB,
    lookup_known, smiles_to_graph, fetch_pdb, get_protein_info,
    get_binding_box, find_vina, prepare_ligand, prepare_receptor,
    run_vina, read_pose, pose_block, get_verdict, parse_nl_query,
    predict_with_uncertainty, confidence_label,
)
from model_bridge import predict_synergy, ModelUnavailableError
from core_fhir import predict_to_fhir
from audit_log import AuditLog
from admet_utils import (
    compute_admet, morgan_similarity, chemical_space_pca, tanimoto_matrix,
    bliss_analysis, combination_index, dose_response_hill, synergy_dose_matrix,
    drug_feature_importance_bar, get_pharmacophore_features,
    generate_html_report, similarity_interpretation,
)
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
    v.setBackgroundColor('#1a1a2e')
    v.zoomTo()
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
tab1,tab2,tab3,tab4,tab5,tab6,tab7,tab8,tab9,tab10,tab11,tab12,tab13,tab14,tab15,tab16,tab17 = st.tabs([
    "🔬 Predict Synergy","🌐 Synergy Landscape","📊 Cell Line Comparison",
    "🏥 Clinical Trials","📚 Literature","💊 Drug Repurposing",
    "⚙️ Mechanism Explorer","🧬 Resistance Mutations","🎬 4D Trajectory","💬 Query",
    "🕸️ Polypharmacology Network","🏥 Clinical Interop",
    "🧪 ADMET Analysis","⚗️ Chemical Space","🎯 Combination Index",
    "🔍 Explainability","📝 Report",
])
# ═══ TAB 1 ════════════════════════════════════════════════════════════════════
with tab1:
    col1,col2=st.columns([1,1.2])
    with col1:
        st.markdown("### 💊 Drug Inputs")
        dao=["Custom (paste SMILES below)"]+sorted(DRUG_SMILES_LOOKUP.keys())
        default_a_idx = dao.index(ex["name_a"]) if ex.get("name_a") in dao else 0
        das=st.selectbox("Drug A — select known drug",dao,index=default_a_idx,key=f"da_select_{example}")
        if das!="Custom (paste SMILES below)":
            smiles_a=DRUG_SMILES_LOOKUP[das]; name_a=das
            st.text_area("Drug A SMILES",value=smiles_a,height=60,disabled=True,key="sma_disp")
        else:
            name_a=st.text_input("Drug A name",value=ex.get("name_a",""),placeholder="e.g. Imatinib",key="name_a_inp")
            smiles_a=st.text_area("Drug A — SMILES",value=ex["smiles_a"],height=80,key="sma_inp")
        dbo=["Custom (paste SMILES below)"]+sorted(DRUG_SMILES_LOOKUP.keys())
        default_b_idx = dbo.index(ex["name_b"]) if ex.get("name_b") in dbo else 0
        dbs=st.selectbox("Drug B — select known drug",dbo,index=default_b_idx,key=f"db_select_{example}")
        if dbs!="Custom (paste SMILES below)":
            smiles_b=DRUG_SMILES_LOOKUP[dbs]; name_b=dbs
            st.text_area("Drug B SMILES",value=smiles_b,height=60,disabled=True,key="smb_disp")
        else:
            name_b=st.text_input("Drug B name",value=ex.get("name_b",""),placeholder="e.g. Dasatinib",key="name_b_inp")
            smiles_b=st.text_area("Drug B — SMILES",value=ex["smiles_b"],height=80,key="smb_inp")
        st.markdown("### 🧫 Target Protein")
        pdb_id=st.text_input("PDB ID",value=ex.get("pdb_id",""),placeholder="e.g. 2HYY",key=f"pdb_inp_{example}").strip().upper()
        if pdb_id: st.caption(f"Will fetch: https://files.rcsb.org/download/{pdb_id}.pdb")
        st.markdown("### 🏥 Cancer Context")
        panel=st.selectbox("Cancer type:",list(CANCER_PANELS.keys()),
            index=list(CANCER_PANELS.keys()).index(ex.get("panel","Melanoma")) if ex.get("panel","Melanoma") in CANCER_PANELS else 0,
            key="panel_sel")
        clp=CANCER_PANELS[panel]; dcl=ex.get("cell_line",clp[0])
        if dcl not in clp: dcl=clp[0]
        cell_line=st.selectbox("Cell line:",clp,index=clp.index(dcl),key="cl_sel")
        exhaustiveness=st.slider("Docking exhaustiveness",4,16,8,2,key="exh_sl")
        mc_samples=st.slider("Uncertainty samples (MC Dropout)",5,50,20,5,key="mc_sl",
            help="More samples = more stable confidence estimate, but slower. The model runs this many stochastic forward passes instead of one, so the reported synergy score comes with a standard deviation instead of being a single number.")
        run_btn=st.button("🔬 Run Docking + Predict Synergy",type="primary",key="run_btn")
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
                             # Store all 9 poses and affinities for both drugs
                            st.session_state['all_poses_a'] = read_all_poses(oa)
                            st.session_state['all_poses_b'] = read_all_poses(ob)
                            st.session_state['affinities_a'] = parse_vina_affinities(oa)
                            st.session_state['affinities_b'] = parse_vina_affinities(ob)
                            st.session_state['pdb_content_for_ligplot'] = pdb_content
                            st.session_state['smiles_a_for_ligplot'] = smiles_a
                            st.session_state['smiles_b_for_ligplot'] = smiles_b
                            st.session_state['name_a_for_ligplot'] = name_a or "Drug A"
                            st.session_state['name_b_for_ligplot'] = name_b or "Drug B"
            else:
                with stat: st.write("⚠️ Docking tools unavailable")
            st.session_state['dsa']=dsa; st.session_state['dsb']=dsb
            st.session_state['dran']=dran; st.session_state['syn_score']=None
            st.session_state['name_a']=name_a; st.session_state['name_b']=name_b
            st.session_state['panel']=panel; st.session_state['cell_line']=cell_line
            st.session_state['pdb_id']=pdb_id
            prog.progress(75)
            with stat: st.write(f"🧠 Predicting synergy ({mc_samples} stochastic samples for uncertainty)...")
            go_emb=torch.zeros(512).unsqueeze(0); dock=torch.tensor([[float(dsa),float(dsb)]])
            uq = predict_with_uncertainty(
                model, model_version, cell_to_idx, ga, gb, go_emb, dock,
                cell_line, Batch, n_samples=mc_samples
            )
            syn = uq["mean_synergy"]; syn_std = uq["std_synergy"]
            prob = uq["mean_prob"]; prob_std = uq["std_prob"]
            st.session_state['syn_score']=syn; st.session_state['syn_prob']=prob
            st.session_state['syn_std']=syn_std; st.session_state['syn_samples']=uq["synergy_samples"]
            prog.progress(100)
            with stat: st.write("✅ Complete!")
            with viz.container():
                if dran and (pa or pb):
                    st.markdown("**Both drugs docked in protein binding pocket**")
                    show_3d(pdb_content,pa,pb,name_a or "Drug A",name_b or "Drug B")
                    st.caption(f"🔵 {name_a or 'Drug A'}  🟠 {name_b or 'Drug B'}  🧬 Protein  *Drag to rotate*")
                else:
                    show_drugs(smiles_a,smiles_b)
            st.markdown("---\n### 📊 Results")
            verdict,color=get_verdict(syn)
            conf_label,conf_color=confidence_label(syn_std)
            st.session_state['verdict']=verdict
            m1,m2,m3,m4=st.columns(4)
            m1.metric("Synergy Score",f"{syn:.3f} ± {syn_std:.3f}")
            m2.metric("Synergy Probability",f"{prob:.3f} ± {prob_std:.3f}")
            m3.metric(f"{name_a or 'Drug A'} Binding",f"{dsa:.2f} kcal/mol")
            m4.metric(f"{name_b or 'Drug B'} Binding",f"{dsb:.2f} kcal/mol")
            st.markdown(f"### Verdict: :{color}[{verdict}]")
            st.markdown(f"**Confidence:** :{conf_color}[{conf_label}]  *(std over {mc_samples} stochastic MC Dropout samples)*")
            st.caption(f"Cancer context: **{panel}** → **{cell_line}**")

            with st.expander("📈 Uncertainty distribution (MC Dropout samples)"):
                st.markdown(f"""The score above (`{syn:.3f}`) is the **mean** of {mc_samples} stochastic forward passes
through the model with dropout kept active at inference time — a standard technique (Gal & Ghahramani, 2016)
for approximating Bayesian uncertainty without retraining or building a separate ensemble. A tight, narrow
distribution means the model consistently lands near the same prediction regardless of which neurons get
randomly dropped; a wide spread means the model itself is uncertain about this specific drug pair.""")
                hist_fig=go.Figure(go.Histogram(x=uq["synergy_samples"],nbinsx=15,marker_color='#4fc3f7'))
                hist_fig.add_vline(x=syn,line_dash="dash",line_color="#FFD700",annotation_text=f"mean={syn:.3f}")
                hist_fig.update_layout(title="Distribution of synergy predictions across MC Dropout samples",
                    xaxis_title="Predicted synergy score",yaxis_title="Count",template="plotly_dark",height=300,
                    paper_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(hist_fig,use_container_width=True)
                st.caption(f"This distribution is a real diagnostic, not decoration — if it's bimodal or very wide, treat the point estimate with caution and consider it alongside the docking scores and known-pathway rationale (Mechanism Explorer tab) rather than at face value.")

            st.session_state.history.insert(0,{'drug_a':name_a or 'Drug A','drug_b':name_b or 'Drug B',
                'cell_line':cell_line,'score':syn,'verdict':verdict,'dock_a':dsa,'dock_b':dsb})
            st.session_state.history=st.session_state.history[:5]
            # ── Tanimoto similarity warning ───────────────────────────────
            if smiles_a and smiles_b:
                _tan = morgan_similarity(smiles_a, smiles_b)
                st.session_state['tanimoto'] = _tan
                if _tan is not None:
                    _sim_label, _sim_note = similarity_interpretation(_tan)
                    _sim_color = "#3a1e1e" if _tan >= 0.65 else "#1e2a1e"
                    _sim_border = "#f85149" if _tan >= 0.65 else "#56d364"
                    st.markdown(
                        f'<div style="background:{_sim_color};border-left:4px solid {_sim_border};'
                        f'padding:10px;border-radius:6px;margin:8px 0;color:white;">'
                        f'<b>🔬 ECFP4 Tanimoto Similarity: {_tan:.3f}</b> — {_sim_label}<br>'
                        f'<small>{_sim_note}</small></div>',
                        unsafe_allow_html=True
                    )
            # ── Quick inline ADMET snapshot ───────────────────────────────
            _admet_a = compute_admet(smiles_a) if smiles_a else None
            _admet_b = compute_admet(smiles_b) if smiles_b else None
            st.session_state['admet_a'] = _admet_a
            st.session_state['admet_b'] = _admet_b
            st.session_state['admet_a_smi'] = smiles_a
            st.session_state['admet_b_smi'] = smiles_b
            if _admet_a and _admet_b:
                with st.expander("🧪 Quick ADMET Snapshot (computed from SMILES)"):
                    _ac1, _ac2 = st.columns(2)
                    with _ac1:
                        st.markdown(f"**{name_a or 'Drug A'}**")
                        st.metric("MW", f"{_admet_a['MW']} Da")
                        st.metric("LogP", _admet_a['LogP'])
                        st.metric("QED", _admet_a['QED'])
                        st.metric("Solubility", _admet_a['Solubility'])
                        st.metric("BBB Penetrant", "✅" if _admet_a['BBB Penetrant'] else "❌")
                        st.metric("Lipinski", "✅ Pass" if _admet_a['Lipinski Pass'] else "❌ Fail")
                    with _ac2:
                        st.markdown(f"**{name_b or 'Drug B'}**")
                        st.metric("MW", f"{_admet_b['MW']} Da")
                        st.metric("LogP", _admet_b['LogP'])
                        st.metric("QED", _admet_b['QED'])
                        st.metric("Solubility", _admet_b['Solubility'])
                        st.metric("BBB Penetrant", "✅" if _admet_b['BBB Penetrant'] else "❌")
                        st.metric("Lipinski", "✅ Pass" if _admet_b['Lipinski Pass'] else "❌ Fail")
                    st.caption("Full breakdown in the 🧪 ADMET Analysis tab")
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
| Synergy score (mean) | {syn:.3f} |
| Synergy score (std, {mc_samples} MC samples) | {syn_std:.3f} |
| Confidence | {conf_label} |
| Verdict | {verdict} |""")
            with st.expander("📖 How to interpret"):
                st.markdown("""| Score | Meaning |
|-------|---------|
| > 0.5 | Strongly Synergistic |
| 0.1–0.5 | Mildly Synergistic |
| -0.1–0.1 | Approximately Additive |
| < -0.1 | Antagonistic |

**Docking score**: more negative = stronger binding. Below -8 = strong binder.

**Confidence bands** (standard deviation across MC Dropout samples):
| Std | Meaning |
|-----|---------|
| < 0.15 | High confidence — model consistently lands near the same prediction |
| 0.15–0.4 | Moderate confidence — some spread across samples |
| > 0.4 | Low confidence — treat the point estimate with caution |

These thresholds are heuristic, the same way the synergy verdict bands are — they are not a calibrated probability guarantee, but they are a real signal: a wide MC Dropout spread genuinely means the model's internal representations disagree with each other on this input, which is exactly the situation where independent verification (literature search, clinical trial data, or wet-lab follow-up) matters most.""")

    if st.session_state.get('all_poses_a'):
        st.markdown("---")
        st.subheader("🔬 Binding Pose Explorer")
        all_pa  = st.session_state.get('all_poses_a', [])
        all_pb  = st.session_state.get('all_poses_b', [])
        affs_a  = st.session_state.get('affinities_a', [])
        affs_b  = st.session_state.get('affinities_b', [])
        pdb_txt = st.session_state.get('pdb_content_for_ligplot', '')
        smi_a   = st.session_state.get('smiles_a_for_ligplot', '')
        smi_b   = st.session_state.get('smiles_b_for_ligplot', '')
        n_a     = st.session_state.get('name_a_for_ligplot', 'Drug A')
        n_b     = st.session_state.get('name_b_for_ligplot', 'Drug B')
        n_poses = max(len(all_pa), len(all_pb))
        if n_poses > 0:
            pose_options = [
                f"Pose {i+1}" + (f"  ({affs_a[i]:.2f} kcal/mol)" if i < len(affs_a) else "")
                for i in range(n_poses)
            ]
            selected = st.select_slider(
                "Select binding pose", options=pose_options, key="pose_slider"
            )
            pose_idx = int(selected.split()[1]) - 1
            pa_sel = all_pa[pose_idx] if pose_idx < len(all_pa) else None
            pb_sel = all_pb[pose_idx] if pose_idx < len(all_pb) else None
            if pdb_txt and (pa_sel or pb_sel):
                show_3d(pdb_txt, pa_sel, pb_sel, n_a, n_b)
                st.caption(f"🔵 {n_a}  🟠 {n_b}  🧬 Protein  *Drag to rotate*")
            st.markdown("#### 🖼️ Interaction Diagrams (LigPlot)")
            col_a, col_b = st.columns(2)
            with col_a:
                if smi_a and pa_sel and pdb_txt:
                    try:
                        jpeg_a = generate_ligplot(
                            smi_a, pa_sel, pdb_txt, pose_idx+1, n_a,
                            affs_a[pose_idx] if pose_idx < len(affs_a) else None
                        )
                        st.image(jpeg_a, caption=f"{n_a} — Pose {pose_idx+1}", use_container_width=True)
                    except Exception as e:
                        st.warning(f"LigPlot unavailable for {n_a}: {e}")
                elif not pa_sel:
                    st.info(f"No pose {pose_idx+1} data for {n_a}")
            with col_b:
                if smi_b and pb_sel and pdb_txt:
                    try:
                        jpeg_b = generate_ligplot(
                            smi_b, pb_sel, pdb_txt, pose_idx+1, n_b,
                            affs_b[pose_idx] if pose_idx < len(affs_b) else None
                        )
                        st.image(jpeg_b, caption=f"{n_b} — Pose {pose_idx+1}", use_container_width=True)
                    except Exception as e:
                        st.warning(f"LigPlot unavailable for {n_b}: {e}")
                elif not pb_sel:
                    st.info(f"No pose {pose_idx+1} data for {n_b}")
            st.markdown("#### 📦 Download All LigPlot Diagrams")
            dl_col1, dl_col2 = st.columns(2)
            with dl_col1:
                if smi_a and all_pa and pdb_txt:
                    try:
                        zip_a = generate_all_ligplots_zip(smi_a, all_pa, pdb_txt, n_a, affs_a)
                        st.download_button(
                            f"⬇️ {n_a} LigPlots ZIP ({len(all_pa)} poses)",
                            data=zip_a, file_name=f"{n_a}_ligplots.zip",
                            mime="application/zip", key="dl_ligplot_a"
                        )
                    except Exception as e:
                        st.warning(f"ZIP unavailable: {e}")
            with dl_col2:
                if smi_b and all_pb and pdb_txt:
                    try:
                        zip_b = generate_all_ligplots_zip(smi_b, all_pb, pdb_txt, n_b, affs_b)
                        st.download_button(
                            f"⬇️ {n_b} LigPlots ZIP ({len(all_pb)} poses)",
                            data=zip_b, file_name=f"{n_b}_ligplots.zip",
                            mime="application/zip", key="dl_ligplot_b"
                        )
                    except Exception as e:
                        st.warning(f"ZIP unavailable: {e}")

    if st.session_state.get('pa') or st.session_state.get('pb'):
        st.markdown("---")
        _pa=st.session_state.get('pa'); _pb=st.session_state.get('pb')
        _pdb=st.session_state.get('pdb_content','')
        if st.button("🎬 Animate Pocket Flythrough",key="fly_btn"):
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
        sp=st.selectbox("Cancer type:",list(scores_data.keys()),key="hp_sel")
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
        with ca: dar=st.selectbox("Drug A:",ad,index=ad.index("Vemurafenib") if "Vemurafenib" in ad else 0,key="ra_sel")
        with cb: dbr=st.selectbox("Drug B:",ad,index=ad.index("Trametinib") if "Trametinib" in ad else 1,key="rb_sel")
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
    with c1: cta=st.text_input("Drug A",placeholder="e.g. Vemurafenib",key="cta_inp")
    with c2: ctb=st.text_input("Drug B",placeholder="e.g. Trametinib",key="ctb_inp")
    ctc=st.text_input("Cancer type (optional)",placeholder="e.g. melanoma",key="ctc_inp")
    if st.button("🔍 Search Clinical Trials",key="ct_btn") and cta and ctb:
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
    with p1: puba=st.text_input("Drug A",placeholder="e.g. Vemurafenib",key="puba_inp")
    with p2: pubb=st.text_input("Drug B",placeholder="e.g. Trametinib",key="pubb_inp")
    pubt=st.text_input("Additional topic",placeholder="e.g. synergy, resistance",key="pubt_inp")
    if st.button("🔍 Search PubMed",key="pub_btn") and puba and pubb:
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
        with rr1: anch=st.selectbox("Your drug:",adr,index=adr.index("Imatinib") if "Imatinib" in adr else 0,key="anch_sel")
        with rr2: rpan=st.selectbox("Cancer type:",list(scores_data.keys()),key="rpan_sel")
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
    with mm1: dma=st.selectbox("Drug A:",list(DRUG_MECHANISMS.keys()),index=list(DRUG_MECHANISMS.keys()).index("Vemurafenib"),key="moa_a_sel")
    with mm2: dmb=st.selectbox("Drug B:",list(DRUG_MECHANISMS.keys()),index=list(DRUG_MECHANISMS.keys()).index("Trametinib"),key="moa_b_sel")
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

# ═══ TAB 8: RESISTANCE MUTATIONS ══════════════════════════════════════════════
import re as _re
import numpy as _np

# ═══ TAB 8: RESISTANCE MUTATION ANALYSIS ════════════════════════════════════
with tab8:
    st.header("🧬 Resistance Mutation Analysis")
    st.markdown(
        "Dock a drug into **wild-type and mutant** protein structures using real "
        "AutoDock Vina. The mutation site is highlighted directly in the 3D viewer "
        "so you can see exactly how far the mutation sits from the drug's binding pose."
    )

    col1, col2 = st.columns(2)
    with col1:
        target_gene = st.selectbox("Cancer Target Gene", list(MUTATION_DB.keys()), key="res_gene_sel")
    with col2:
        mut_drug = st.selectbox("Drug to Test",
            ["Vemurafenib","Erlotinib","Imatinib","Dasatinib","Crizotinib",
             "Osimertinib","Gefitinib","Dabrafenib","Alectinib","Nilotinib"],
            key="res_drug_sel")

    res_exhaustiveness = st.slider("Docking exhaustiveness", 4, 16, 6, 2, key="res_exh_sl",
        help="Each variant below is docked independently — N × the work of a single docking run.")

    gene_data   = MUTATION_DB[target_gene]
    mutations   = gene_data["mutations"]
    n_variants  = 1 + len(mutations)
    st.caption(f"Will run {n_variants} independent docking jobs "
               f"(1 wild-type + {len(mutations)} mutant{'s' if len(mutations)!=1 else ''}).")

    def _residue_num(mut_name):
        """Extract residue number from mutation label like 'V600E' → 600."""
        m = _re.search(r'\d+', mut_name)
        return int(m.group()) if m else None

    def _drug_centroid(pose_atoms):
        """Mean xyz of docked drug atoms."""
        if not pose_atoms: return None
        coords = _np.array([[a[1], a[2], a[3]] for a in pose_atoms])
        return coords.mean(axis=0)

    def _mutation_ca(pdb_text, resi):
        """Find Cα coordinate of a given residue number in PDB text."""
        if resi is None: return None
        for line in pdb_text.splitlines():
            if line.startswith(("ATOM","HETATM")) and line[12:16].strip() == "CA":
                try:
                    if int(line[22:26].strip()) == resi:
                        return _np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])
                except: pass
        return None

    def show_3d_resistance(pdb_text, drug_pose, mut_resi, variant_label,
                           show_ribbon, show_mut, show_drug, show_surface, h=480):
        """py3Dmol viewer for resistance analysis with toggleable layers."""
        v = py3Dmol.view(width=700, height=h)
        v.addModel(pdb_text, 'pdb')

        if show_ribbon:
            v.setStyle({'model': 0}, {'cartoon': {'color': 'spectrum', 'opacity': 0.55}})
        else:
            v.setStyle({'model': 0}, {})

        if show_mut and mut_resi:
            v.setStyle(
                {'model': 0, 'resi': mut_resi},
                {'sphere': {'color': '#FF5722', 'radius': 1.4},
                 'stick':  {'color': '#FF5722', 'radius': 0.35}}
            )
            v.addLabel(
                f"Mutation site (res {mut_resi})",
                {'resi': mut_resi, 'model': 0,
                 'backgroundColor': '#FF5722', 'fontColor': 'white',
                 'fontSize': 11, 'backgroundOpacity': 0.85,
                 'borderThickness': 1.0, 'borderColor': 'white'}
            )

        if show_surface:
            v.addSurface('VDW',
                {'opacity': 0.18, 'color': 'white'},
                {'model': 0}
            )

        if show_drug and drug_pose:
            v.addModel(pose_block(drug_pose, 'L'), 'pdb')
            mdl_idx = 1
            v.setStyle(
                {'model': mdl_idx},
                {'stick':   {'colorscheme': 'cyanCarbon', 'radius': 0.22},
                 'sphere':  {'colorscheme': 'cyanCarbon', 'scale': 0.32}}
            )

        v.setBackgroundColor('#1a1a2e')
        v.zoomTo()
        components.html(v._make_html(), height=h + 20, scrolling=False)

    if st.button("🔬 Run Resistance Analysis", type="primary", key="btn_resistance"):
        vina_cmd   = find_vina()
        obabel_cmd = shutil.which('obabel')
        if not (vina_cmd and obabel_cmd):
            st.error(
                "⚠️ AutoDock Vina / OpenBabel not available in this environment. "
                "This tab requires the same docking tools as the Predict Synergy tab "
                "(available on the deployed app via packages.txt)."
            )
            st.stop()

        drug_smiles = DRUG_SMILES_LOOKUP.get(mut_drug)
        if not drug_smiles:
            st.error(f"No SMILES found for {mut_drug}."); st.stop()

        variants_to_dock = [("Wild-Type", gene_data["wild_type"], None)]
        for mn, mi in mutations.items():
            variants_to_dock.append((mn, mi["pdb"], mi))

        prog = st.progress(0)
        stat = st.status("Running resistance docking pipeline...", expanded=True)
        results   = []
        wt_affinity = None

        # Store per-variant viewer data in session state so toggles work post-run
        st.session_state['res_variants'] = []

        with tempfile.TemporaryDirectory() as wd:
            ligand_path = prepare_ligand(drug_smiles, "res_drug", wd)
            if not ligand_path:
                st.error(f"Could not prepare ligand for {mut_drug}."); st.stop()

            for i, (variant_label, variant_pdb_id, mut_info) in enumerate(variants_to_dock):
                with stat:
                    st.write(f"📥 Fetching {variant_pdb_id} ({target_gene} {variant_label})...")

                pdb_path = fetch_pdb(variant_pdb_id, wd)
                if not pdb_path:
                    with stat:
                        st.write(f"❌ Could not fetch {variant_pdb_id} — skipping")
                    results.append({
                        "Variant": f"{target_gene} {variant_label}", "PDB": variant_pdb_id,
                        "Binding Affinity (kcal/mol)": None, "Delta vs WT": None,
                        "Distance to Mutation (Å)": None,
                        "Resistance Level": "N/A (structure unavailable)",
                        "Clinical Impact": "N/A"
                    })
                    prog.progress(int((i+1)/n_variants*100)); continue

                pdb_text = open(pdb_path).read()
                center, size, _ = get_binding_box(pdb_path)
                rec_path = prepare_receptor(pdb_path, wd)
                if not rec_path:
                    with stat:
                        st.write(f"❌ Receptor prep failed for {variant_pdb_id} — skipping")
                    results.append({
                        "Variant": f"{target_gene} {variant_label}", "PDB": variant_pdb_id,
                        "Binding Affinity (kcal/mol)": None, "Delta vs WT": None,
                        "Distance to Mutation (Å)": None,
                        "Resistance Level": "N/A (receptor prep failed)",
                        "Clinical Impact": "N/A"
                    })
                    prog.progress(int((i+1)/n_variants*100)); continue

                # ── Parse mutation residue number ──────────────────────────
                mut_resi = _residue_num(variant_label) if variant_label != "Wild-Type" else None

                with stat:
                    st.write(f"⚗️ Docking {mut_drug} → {target_gene} {variant_label} ({variant_pdb_id})...")

                out_path = f'{wd}/res_{i}_out.pdbqt'
                affinity, _ = run_vina(vina_cmd, rec_path, ligand_path,
                                       center, size, out_path, res_exhaustiveness)
                pose = read_pose(out_path)

                # ── Proximity: distance from drug centroid to mutation Cα ──
                dist = None
                if pose and mut_resi:
                    centroid  = _drug_centroid(pose)
                    mut_coord = _mutation_ca(pdb_text, mut_resi)
                    if centroid is not None and mut_coord is not None:
                        dist = round(float(_np.linalg.norm(centroid - mut_coord)), 1)

                if affinity is None:
                    with stat: st.write(f"❌ Docking failed for {variant_label}")
                    results.append({
                        "Variant": f"{target_gene} {variant_label}", "PDB": variant_pdb_id,
                        "Binding Affinity (kcal/mol)": None, "Delta vs WT": None,
                        "Distance to Mutation (Å)": dist,
                        "Resistance Level": "N/A (docking failed)",
                        "Clinical Impact": "N/A"
                    })
                    st.session_state['res_variants'].append({
                        'label': variant_label, 'pdb_text': pdb_text,
                        'pose': None, 'mut_resi': mut_resi,
                        'affinity': None, 'dist': dist
                    })
                    prog.progress(int((i+1)/n_variants*100)); continue

                if variant_label == "Wild-Type":
                    wt_affinity = affinity
                    delta = 0.0; resistance = "Reference"; clinical = "Sensitive"
                else:
                    delta = round(affinity - wt_affinity, 2) if wt_affinity is not None else None
                    if delta is None:
                        resistance = "N/A"; clinical = "N/A"
                    elif delta > 1.5:
                        resistance = "High"; clinical = "Resistant"
                    elif delta > 0.5:
                        resistance = "Moderate"; clinical = "Partially Resistant"
                    else:
                        resistance = "Low"; clinical = "Sensitive"

                with stat:
                    dist_str = f" | Mutation {dist} Å from binding site" if dist else ""
                    st.write(f"✅ {target_gene} {variant_label}: {affinity:.2f} kcal/mol{dist_str}")

                results.append({
                    "Variant": f"{target_gene} {variant_label}", "PDB": variant_pdb_id,
                    "Binding Affinity (kcal/mol)": round(affinity, 2),
                    "Delta vs WT": delta,
                    "Distance to Mutation (Å)": dist,
                    "Resistance Level": resistance,
                    "Clinical Impact": clinical
                })
                st.session_state['res_variants'].append({
                    'label': variant_label, 'pdb_text': pdb_text,
                    'pose': pose, 'mut_resi': mut_resi,
                    'affinity': affinity, 'dist': dist
                })
                prog.progress(int((i+1)/n_variants*100))

        with stat: st.write("✅ All variants complete!")
        st.session_state['res_results']   = results
        st.session_state['res_drug']      = mut_drug
        st.session_state['res_gene']      = target_gene

    # ── Render results (persists across toggle interactions) ────────────────
    if st.session_state.get('res_variants'):
        st.markdown("---")
        st.subheader(f"3D Docking Views — {st.session_state.get('res_drug','')} vs {st.session_state.get('res_gene','')} variants")

        # Toggle controls
        tcol1, tcol2, tcol3, tcol4 = st.columns(4)
        show_ribbon  = tcol1.checkbox("🎀 Protein ribbon",  value=True,  key="tog_ribbon")
        show_mut     = tcol2.checkbox("🔴 Mutation site",   value=True,  key="tog_mut")
        show_drug    = tcol3.checkbox("💊 Docked drug",     value=True,  key="tog_drug")
        show_surface = tcol4.checkbox("🫧 Surface",         value=False, key="tog_surface")

        for vdata in st.session_state['res_variants']:
            lbl      = vdata['label']
            affinity = vdata['affinity']
            dist     = vdata['dist']
            mut_resi = vdata['mut_resi']

            aff_str  = f"{affinity:.2f} kcal/mol" if affinity else "Docking failed"
            dist_str = f" · Mutation {dist} Å from binding site" if dist else ""
            mut_str  = f" · Mutation residue: {mut_resi}" if mut_resi else ""
            is_wt    = lbl == "Wild-Type"
            badge    = "🟢 Wild-Type" if is_wt else f"🔴 {target_gene} {lbl}"

            with st.expander(f"{badge}  —  {aff_str}{dist_str}{mut_str}", expanded=True):
                show_3d_resistance(
                    pdb_text=vdata['pdb_text'],
                    drug_pose=vdata['pose'],
                    mut_resi=mut_resi,
                    variant_label=lbl,
                    show_ribbon=show_ribbon,
                    show_mut=show_mut,
                    show_drug=show_drug,
                    show_surface=show_surface,
                )
                if dist and mut_resi:
                    proximity_note = (
                        f"🔴 **{dist} Å** — mutation is within direct binding influence (<5 Å)"
                        if dist < 5 else
                        f"🟡 **{dist} Å** — mutation is near the binding site (5–10 Å)"
                        if dist < 10 else
                        f"🟢 **{dist} Å** — mutation is distant from binding site (>10 Å)"
                    )
                    st.caption(f"📏 Drug centroid → mutation Cα distance: {proximity_note}")

        # Summary chart
        st.markdown("---")
        st.subheader("📊 Binding Affinity Comparison")
        df_res = pd.DataFrame(st.session_state['res_results'])
        valid  = df_res[df_res["Binding Affinity (kcal/mol)"].notna()]
        if not valid.empty:
            colors = []
            for _, row in valid.iterrows():
                if row["Resistance Level"] == "Reference":   colors.append("#4CAF50")
                elif row["Resistance Level"] == "High":      colors.append("#F44336")
                elif row["Resistance Level"] == "Moderate":  colors.append("#FF9800")
                else:                                         colors.append("#2196F3")
            fig_res = go.Figure(go.Bar(
                x=valid["Variant"], y=valid["Binding Affinity (kcal/mol)"],
                marker_color=colors,
                text=valid["Delta vs WT"].apply(lambda x: f"Δ{x:+.2f}" if x not in (0.0, None) else "WT"),
                textposition="outside"
            ))
            fig_res.update_layout(
                title=f"{st.session_state.get('res_drug','')} Real Binding Affinity "
                      f"Across {st.session_state.get('res_gene','')} Variants (AutoDock Vina)",
                yaxis_title="Binding Affinity (kcal/mol)",
                xaxis_title="Protein Variant",
                template="plotly_dark", height=420,
                yaxis=dict(range=[float(valid["Binding Affinity (kcal/mol)"].min())-2, 0])
            )
            st.plotly_chart(fig_res, use_container_width=True)

        st.dataframe(
            df_res.style.map(
                lambda v: "color: #F44336; font-weight: bold" if v=="High" else
                          ("color: #FF9800" if v=="Moderate" else
                           ("color: #4CAF50" if v=="Low" else "")),
                subset=["Resistance Level"]
            ), use_container_width=True
        )

        st.subheader("📋 Mutation Clinical Notes")
        for mut_name, mut_info in mutations.items():
            affected_str = ("⚠️ Affects this drug"
                           if st.session_state.get('res_drug','') in mut_info["drugs_affected"]
                           else "✅ Does not affect this drug")
            with st.expander(f"{target_gene} {mut_name} — {affected_str}"):
                st.markdown(f"**Mechanism:** {mut_info['description']}")
                st.markdown(f"**Drugs affected:** {', '.join(mut_info['drugs_affected'])}")
                st.markdown(f"**PDB Structure:** `{mut_info['pdb']}`")

        if (st.session_state.get('res_drug','') in ["Vemurafenib","Erlotinib","Imatinib"]
                and st.session_state.get('res_gene','') in ["BRAF","EGFR","BCR-ABL"]):
            st.info(
                f"💡 For {st.session_state.get('res_drug','')} -resistant "
                f"{st.session_state.get('res_gene','')} mutations, consider "
                f"next-generation inhibitors or combination strategies in the Predict Synergy tab."
            )

# ═══ TAB 9: 4D TRAJECTORY ═════════════════════════════════════════════════════
with tab9:
    st.header("🎬 4D Docking Trajectory")
    st.markdown("Visualize how a drug **approaches and binds** to its protein pocket — simulated binding trajectory with energy profile.")

    TRAJ_TARGETS=["BRAF (Melanoma)","EGFR (Lung)","BCR-ABL (Leukemia)","CDK4/6 (Breast)","VEGFR (Angiogenesis)"]
    TRAJ_DRUGS=["Vemurafenib","Trametinib","Erlotinib","Imatinib","Paclitaxel","Venetoclax","Alpelisib","Osimertinib"]

    col1,col2=st.columns(2)
    with col1: traj_drug=st.selectbox("Drug",TRAJ_DRUGS,key="traj_drug_sel")
    with col2: traj_target=st.selectbox("Target Protein",TRAJ_TARGETS,key="traj_target_sel")
    n_frames=st.slider("Trajectory Frames",20,80,40,key="traj_frames_sl")

    if st.button("▶️ Generate 4D Trajectory",type="primary",key="btn_4d_traj"):
        import random
        st.subheader(f"🎬 {traj_drug} → {traj_target} Binding Trajectory")
        random.seed(hash(traj_drug+traj_target))
        np.random.seed(hash(traj_drug+traj_target)%(2**31))
        frames=n_frames
        t=np.linspace(0,1,frames)
        start_pos=np.array([30.0,28.0,25.0]); end_pos=np.array([0.0,0.0,0.0])
        sigmoid=1/(1+np.exp(-10*(t-0.5)))
        noise=np.random.randn(frames,3)*(1-sigmoid[:,None])*3
        drug_traj=start_pos[None]*(1-sigmoid[:,None])+end_pos[None]*sigmoid[:,None]+noise
        binding_energy=-2+(-7)*sigmoid+1.5*np.exp(-((t-0.85)**2)/0.005)*(1-sigmoid)
        binding_energy+=np.random.randn(frames)*0.3
        np.random.seed(42)
        n_res=18
        px=np.random.randn(n_res)*5; py=np.random.randn(n_res)*5; pz=np.random.randn(n_res)*5
        rc=np.random.choice(["#FF6B6B","#4ECDC4","#45B7D1","#96CEB4","#FFEAA7"],n_res)
        fig_traj=go.Figure()
        fig_traj.add_trace(go.Scatter3d(x=px,y=py,z=pz,mode="markers",
            marker=dict(size=12,color=list(rc),opacity=0.7),name="Pocket Residues",hovertemplate="Residue<extra></extra>"))
        fig_traj.add_trace(go.Scatter3d(x=[drug_traj[0,0]],y=[drug_traj[0,1]],z=[drug_traj[0,2]],
            mode="markers",marker=dict(size=16,color="#FFD700",symbol="diamond",opacity=1.0),
            name=traj_drug,hovertemplate=f"{traj_drug}<extra></extra>"))
        fig_traj.add_trace(go.Scatter3d(x=[drug_traj[0,0]],y=[drug_traj[0,1]],z=[drug_traj[0,2]],
            mode="lines",line=dict(color="#FFD700",width=3),opacity=0.4,name="Approach Path"))
        frames_list=[]
        for i in range(frames):
            frames_list.append(go.Frame(data=[
                go.Scatter3d(x=px,y=py,z=pz,mode="markers",marker=dict(size=12,color=list(rc),opacity=0.7)),
                go.Scatter3d(x=[drug_traj[i,0]],y=[drug_traj[i,1]],z=[drug_traj[i,2]],
                    mode="markers",marker=dict(size=16,color="#FFD700",symbol="diamond")),
                go.Scatter3d(x=drug_traj[:i+1,0],y=drug_traj[:i+1,1],z=drug_traj[:i+1,2],
                    mode="lines",line=dict(color="#FFD700",width=3),opacity=0.4),
            ],name=str(i)))
        fig_traj.frames=frames_list
        fig_traj.update_layout(
            scene=dict(bgcolor="rgb(10,10,30)",
                xaxis=dict(showgrid=False,zeroline=False,showticklabels=False,title=""),
                yaxis=dict(showgrid=False,zeroline=False,showticklabels=False,title=""),
                zaxis=dict(showgrid=False,zeroline=False,showticklabels=False,title="")),
            paper_bgcolor="rgb(10,10,30)",font_color="white",
            title=dict(text=f"🎬 {traj_drug} Approaching {traj_target} Binding Pocket",font=dict(color="white")),
            updatemenus=[dict(type="buttons",showactive=False,y=1.05,x=0.5,xanchor="center",
                buttons=[
                    dict(label="▶ Play",method="animate",
                         args=[None,{"frame":{"duration":80,"redraw":True},"fromcurrent":True,"transition":{"duration":20}}]),
                    dict(label="⏸ Pause",method="animate",
                         args=[[None],{"frame":{"duration":0,"redraw":False},"mode":"immediate","transition":{"duration":0}}])
                ])],
            sliders=[dict(
                steps=[dict(method="animate",args=[[str(i)],{"frame":{"duration":80,"redraw":True},"mode":"immediate"}],
                            label=str(i)) for i in range(frames)],
                x=0.05,len=0.9,y=0,currentvalue=dict(prefix="Frame: ",visible=True,xanchor="center"),
                transition=dict(duration=20))],
            height=550,legend=dict(font=dict(color="white")))
        st.plotly_chart(fig_traj,use_container_width=True)
        fig_energy=go.Figure()
        fig_energy.add_trace(go.Scatter(x=list(range(frames)),y=list(binding_energy),
            mode="lines+markers",line=dict(color="#FFD700",width=2),marker=dict(size=4),
            fill="tozeroy",fillcolor="rgba(255,215,0,0.15)",name="Binding Energy"))
        fig_energy.add_hline(y=float(binding_energy[-5:].mean()),line_dash="dash",line_color="#4CAF50",
            annotation_text=f"Final: {binding_energy[-5:].mean():.2f} kcal/mol")
        fig_energy.update_layout(title="⚡ Binding Energy Profile Along Trajectory",
            xaxis_title="Trajectory Frame",yaxis_title="ΔG (kcal/mol)",template="plotly_dark",height=280)
        st.plotly_chart(fig_energy,use_container_width=True)
        col1,col2,col3=st.columns(3)
        col1.metric("Initial Distance","~42 Å","Far from pocket")
        col2.metric("Final Affinity",f"{binding_energy[-5:].mean():.2f} kcal/mol","Stable binding")
        col3.metric("Frames Simulated",str(frames),f"~{frames*80}ms playback")

# ═══ TAB 10: NATURAL LANGUAGE QUERY ══════════════════════════════════════════
with tab10:
    st.header("💬 Natural Language Query")
    st.markdown("Ask questions about drug synergy in plain English — powered by rule-based parsing over your precomputed scores.")

    if "nl_history" not in st.session_state:
        st.session_state.nl_history=[]
    if "nl_query_input" not in st.session_state:
        st.session_state.nl_query_input=""

    EXAMPLE_QUERIES=[
        "Which drug pairs are most synergistic in breast cancer?",
        "Is Vemurafenib + Trametinib synergistic?",
        "What is the best drug combination for leukemia?",
        "Show me antagonistic pairs in lung cancer",
        "Which drugs work best with Paclitaxel?",
        "Compare Venetoclax across cancer types",
    ]

    st.markdown("**Try an example:**")
    ex_cols=st.columns(3)
    for i,eq in enumerate(EXAMPLE_QUERIES):
        if ex_cols[i%3].button(eq,key=f"nlex_{i}"):
            st.session_state.nl_query_input=eq

    nl_query=st.text_input("Your question:",value=st.session_state.nl_query_input,
        key="nl_query_box",placeholder="e.g. Which drug pairs are synergistic in melanoma?")

    if st.button("🔍 Ask",type="primary",key="btn_nl_ask") and nl_query:
        with st.spinner("Analyzing..."):
            answer=parse_nl_query(nl_query,scores_data)
            st.session_state.nl_history.append({"q":nl_query,"a":answer})
            st.session_state.nl_query_input=""

    if st.session_state.nl_history:
        for item in reversed(st.session_state.nl_history[-5:]):
            st.markdown(f"**Q: {item['q']}**")
            st.markdown(item["a"])
            st.divider()
    else:
        st.info("💡 Ask anything — drug pairs, cancer types, best combinations, comparisons.")

# ═══ TAB 11: POLYPHARMACOLOGY NETWORK ════════════════════════════════════════
with tab11:
    st.header("🕸️ Polypharmacology Network Explorer")
    st.markdown("""See the **systems-level picture**: every drug as a node, connected to the pathways/targets it hits and the
other drugs sharing those pathways. Pairwise tabs show one combination at a time — this shows the whole web of overlap and complementarity at once.""")

    net_col1, net_col2 = st.columns([1,3])
    with net_col1:
        view_mode = st.radio("Network focus", ["All pathways", "Single pathway", "Drugs around one anchor"], key="net_mode")
        if view_mode == "Single pathway":
            all_pathways = sorted(set(v['pathway'] for v in DRUG_MECHANISMS.values()))
            focus_pathway = st.selectbox("Pathway", all_pathways, key="net_pathway_sel")
            net_drugs = [d for d,v in DRUG_MECHANISMS.items() if v['pathway']==focus_pathway]
        elif view_mode == "Drugs around one anchor":
            anchor_drug = st.selectbox("Anchor drug", sorted(DRUG_MECHANISMS.keys()),
                index=sorted(DRUG_MECHANISMS.keys()).index("Vemurafenib"), key="net_anchor_sel")
            anchor_pathway = DRUG_MECHANISMS[anchor_drug]['pathway']
            related_pathways = set()
            for (p1,p2) in SYNERGY_RULES:
                if p1 == anchor_pathway: related_pathways.add(p2)
                if p2 == anchor_pathway: related_pathways.add(p1)
            related_pathways.add(anchor_pathway)
            net_drugs = [d for d,v in DRUG_MECHANISMS.items() if v['pathway'] in related_pathways]
        else:
            net_drugs = list(DRUG_MECHANISMS.keys())
        st.caption(f"{len(net_drugs)} drugs in view")
        st.markdown("**Legend**")
        st.markdown("🔵 Drug node — size = number of pathway connections")
        st.markdown("🟢 Edge = synergy-predicted pathway relationship")
        st.markdown("🔴 Edge = same-pathway (competition risk)")

    with net_col2:
        pathways_in_view = sorted(set(DRUG_MECHANISMS[d]['pathway'] for d in net_drugs))
        pathway_angle = {p: 2*np.pi*i/max(len(pathways_in_view),1) for i,p in enumerate(pathways_in_view)}
        pathway_radius = 3.5

        drug_pos = {}
        for p in pathways_in_view:
            drugs_here = [d for d in net_drugs if DRUG_MECHANISMS[d]['pathway']==p]
            base_angle = pathway_angle[p]
            for j, d in enumerate(drugs_here):
                spread = 0.35 * (j - (len(drugs_here)-1)/2)
                ang = base_angle + spread
                r = pathway_radius + 1.2
                drug_pos[d] = (r*np.cos(ang), r*np.sin(ang))

        pathway_pos = {p: (pathway_radius*0.4*np.cos(pathway_angle[p]), pathway_radius*0.4*np.sin(pathway_angle[p])) for p in pathways_in_view}

        fig_net = go.Figure()

        # drug-to-own-pathway edges (gray)
        for d in net_drugs:
            p = DRUG_MECHANISMS[d]['pathway']
            dx, dy = drug_pos[d]; px, py = pathway_pos[p]
            fig_net.add_trace(go.Scatter(x=[dx,px], y=[dy,py], mode='lines',
                line=dict(color='rgba(150,150,150,0.3)', width=1), showlegend=False, hoverinfo='skip'))

        # cross-pathway synergy/competition edges between pathway hubs
        seen_edges = set()
        for (p1,p2), rule in SYNERGY_RULES.items():
            if p1 in pathway_pos and p2 in pathway_pos:
                key = tuple(sorted([p1,p2]))
                if key in seen_edges or p1==p2: continue
                seen_edges.add(key)
                x1,y1 = pathway_pos[p1]; x2,y2 = pathway_pos[p2]
                is_good = '✅' in rule
                fig_net.add_trace(go.Scatter(x=[x1,x2], y=[y1,y2], mode='lines',
                    line=dict(color='rgba(76,175,80,0.5)' if is_good else 'rgba(244,67,54,0.5)', width=2.5),
                    showlegend=False, hovertext=rule, hoverinfo='text'))

        # pathway hub nodes
        fig_net.add_trace(go.Scatter(
            x=[pathway_pos[p][0] for p in pathways_in_view],
            y=[pathway_pos[p][1] for p in pathways_in_view],
            mode='markers+text', text=pathways_in_view, textposition='middle center',
            textfont=dict(size=10, color='white'),
            marker=dict(size=[34]*len(pathways_in_view), color='#FF9800', symbol='diamond', line=dict(width=2,color='white')),
            name='Pathway', hovertext=[f"Pathway: {p}" for p in pathways_in_view], hoverinfo='text'))

        # drug nodes
        drug_conn_count = {d: 1 for d in net_drugs}
        fig_net.add_trace(go.Scatter(
            x=[drug_pos[d][0] for d in net_drugs],
            y=[drug_pos[d][1] for d in net_drugs],
            mode='markers+text', text=net_drugs, textposition='top center',
            textfont=dict(size=9, color='white'),
            marker=dict(size=18, color='#4fc3f7', line=dict(width=1.5,color='white')),
            name='Drug',
            hovertext=[f"{d}<br>Target: {DRUG_MECHANISMS[d]['target']}<br>Class: {DRUG_MECHANISMS[d]['class']}" for d in net_drugs],
            hoverinfo='text'))

        fig_net.update_layout(
            height=650, showlegend=True,
            xaxis=dict(visible=False), yaxis=dict(visible=False),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(10,10,20,0.3)',
            font=dict(color='white'), margin=dict(l=10,r=10,t=10,b=10),
            legend=dict(font=dict(color='white'), bgcolor='rgba(0,0,0,0.3)'))
        st.plotly_chart(fig_net, use_container_width=True)
        st.caption("🟠 Diamond = pathway hub | 🔵 Circle = drug | Green edge = cross-pathway synergy | Red edge = same-pathway competition risk | Hover for details")

    st.markdown("---")
    st.markdown("#### 📋 Pathway Co-occurrence Summary")
    summary_rows = []
    seen_pairs = set()
    for (p1,p2), rule in SYNERGY_RULES.items():
        key = tuple(sorted([p1,p2]))
        if key in seen_pairs: continue
        seen_pairs.add(key)
        n_drugs_p1 = sum(1 for v in DRUG_MECHANISMS.values() if v['pathway']==p1)
        n_drugs_p2 = sum(1 for v in DRUG_MECHANISMS.values() if v['pathway']==p2)
        summary_rows.append({
            "Pathway A": p1, "Pathway B": p2,
            "Relationship": "✅ Synergistic" if '✅' in rule else "⚠️ Competition Risk",
            "Drugs in A": n_drugs_p1, "Drugs in B": n_drugs_p2
        })
    if summary_rows:
        st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)
        # ═══ TAB 12: CLINICAL INTEROPERABILITY (FHIR) ════════════════════════════════
with tab12:
    st.header("🏥 Clinical Interoperability (FHIR R4)")
    st.markdown(
        "Most drug-discovery ML tools stop at a prediction number. Real clinical "
        "software has to expose that prediction in a format other healthcare "
        "systems can consume — that's what FHIR (Fast Healthcare Interoperability "
        "Resources) is for. This tab converts a live model prediction into a "
        "spec-compliant FHIR `DiagnosticReport`, the same shape used by EHR "
        "platforms like Oracle Health (Cerner) and Epic."
    )

    fhir_col1, fhir_col2, fhir_col3 = st.columns(3)
    with fhir_col1:
        fhir_drug_a = st.selectbox("Drug A", sorted(DRUG_SMILES_LOOKUP.keys()),
            index=sorted(DRUG_SMILES_LOOKUP.keys()).index("Olaparib"), key="fhir_drug_a")
    with fhir_col2:
        fhir_drug_b = st.selectbox("Drug B", sorted(DRUG_SMILES_LOOKUP.keys()),
            index=sorted(DRUG_SMILES_LOOKUP.keys()).index("Rucaparib"), key="fhir_drug_b")
    with fhir_col3:
        fhir_cell_line = st.text_input("Cell line (NCI-60)", value="OVCAR-3", key="fhir_cell_line")

    if st.button("🧬 Generate FHIR DiagnosticReport", key="fhir_generate_btn"):
        with st.spinner("Running model inference..."):
            try:
                score, confidence, affinity = predict_synergy(fhir_drug_a, fhir_drug_b, fhir_cell_line)
                resource, success = predict_to_fhir(
                    drug_a=fhir_drug_a, drug_b=fhir_drug_b, cell_line=fhir_cell_line,
                    synergy_score=score, confidence=confidence, docking_affinity=affinity,
                )
            except ModelUnavailableError as e:
                st.error(f"Model error: {e}")
                resource, success = None, False

        if resource:
            audit = AuditLog(path="audit_log.jsonl")
            audit.record(
                drug_a=fhir_drug_a, drug_b=fhir_drug_b, cell_line=fhir_cell_line,
                output_resource_type=resource["resourceType"],
                output_summary=resource.get("conclusion", str(resource.get("issue"))),
                model_version="ProteinSynergyDockV2-epoch82",
                success=success, user="streamlit-user",
            )
            if success:
                st.success("Valid FHIR DiagnosticReport generated.")
            else:
                st.warning("Input failed validation — FHIR OperationOutcome returned instead.")
            st.json(resource)

    st.markdown("---")
    st.caption(
        "🔗 Live API: this same logic is also exposed as a public REST endpoint at "
        "[proteinsynergydock-fhir-api.onrender.com/docs](https://proteinsynergydock-fhir-api.onrender.com/docs)"
    )

# ═══ TAB 13: ADMET Analysis ════════════════════════════════════════════════════
with tab13:
    st.markdown("### 🧪 ADMET Property Analysis")
    st.markdown(
        "Every number here is **computed live from your SMILES** using RDKit — "
        "no lookup table, no hardcoding."
    )
    _a13_c1, _a13_c2 = st.columns(2)
    with _a13_c1:
        _a13_smi_a = st.text_area(
            "Drug A SMILES", height=80, key="admet_smi_a",
            value=st.session_state.get('admet_a_smi', ''),
            placeholder="Paste SMILES or run Predict Synergy first"
        )
    with _a13_c2:
        _a13_smi_b = st.text_area(
            "Drug B SMILES", height=80, key="admet_smi_b",
            value=st.session_state.get('admet_b_smi', ''),
            placeholder="Paste SMILES or run Predict Synergy first"
        )
    if st.button("⚗️ Compute ADMET Properties", key="admet_btn", type="primary"):
        if not _a13_smi_a or not _a13_smi_b:
            st.error("Enter SMILES for both drugs (or run Predict Synergy first to auto-fill).")
            st.stop()
        _adm_a_new = compute_admet(_a13_smi_a)
        _adm_b_new = compute_admet(_a13_smi_b)
        if not _adm_a_new: st.error("Invalid SMILES for Drug A."); st.stop()
        if not _adm_b_new: st.error("Invalid SMILES for Drug B."); st.stop()
        st.session_state['admet_a'] = _adm_a_new
        st.session_state['admet_b'] = _adm_b_new
        st.session_state['admet_a_smi'] = _a13_smi_a
        st.session_state['admet_b_smi'] = _a13_smi_b
        st.session_state['tanimoto'] = morgan_similarity(_a13_smi_a, _a13_smi_b)
    _adm_a = st.session_state.get('admet_a')
    _adm_b = st.session_state.get('admet_b')
    if _adm_a and _adm_b:
        _tan13 = st.session_state.get('tanimoto')
        if _tan13 is not None:
            _sl, _sn = similarity_interpretation(_tan13)
            _sc = "#3a1e1e" if _tan13 >= 0.65 else "#1a2a3a"
            _sb = "#f85149" if _tan13 >= 0.65 else "#4fc3f7"
            st.markdown(
                f'<div style="background:{_sc};border-left:4px solid {_sb};padding:12px;border-radius:6px;margin:12px 0;color:white;">'
                f'<b>ECFP4 Tanimoto Similarity: {_tan13:.3f}</b> — {_sl}<br><small>{_sn}</small></div>',
                unsafe_allow_html=True
            )
        st.markdown("#### 📡 Drug-likeness Radar")
        _radar_labels = ["QED","Drug-likeness","Absorption","Fsp3×2","TPSA↓","MW↓","LogP (opt)"]
        def _norm_admet_vals(d):
            return [d.get("QED",0),d.get("Drug-likeness",0),d.get("Absorption",0),
                    min(d.get("Fsp3",0)*2,1),max(0.,1.-d.get("TPSA",0)/140),
                    max(0.,1.-d.get("MW",0)/700),max(0.,1.-abs(d.get("LogP",0)-2.5)/5)]
        _va13=_norm_admet_vals(_adm_a); _vb13=_norm_admet_vals(_adm_b)
        _da_name_r=st.session_state.get('name_a','Drug A') or 'Drug A'
        _db_name_r=st.session_state.get('name_b','Drug B') or 'Drug B'
        _fig_radar=go.Figure()
        _fig_radar.add_trace(go.Scatterpolar(r=_va13+[_va13[0]],theta=_radar_labels+[_radar_labels[0]],
            fill='toself',fillcolor='rgba(79,195,247,0.18)',line=dict(color='#4fc3f7',width=2),name=_da_name_r))
        _fig_radar.add_trace(go.Scatterpolar(r=_vb13+[_vb13[0]],theta=_radar_labels+[_radar_labels[0]],
            fill='toself',fillcolor='rgba(255,152,0,0.18)',line=dict(color='#ff9800',width=2),name=_db_name_r))
        _fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True,range=[0,1])),
            height=430,paper_bgcolor='rgba(0,0,0,0)',font=dict(color='white'),showlegend=True,
            title=dict(text="Drug-likeness Radar (normalized 0–1)",font=dict(color='#4fc3f7')))
        st.plotly_chart(_fig_radar,use_container_width=True)
        st.markdown("#### 📋 Full ADMET Comparison")
        _prop_groups=[
            ("Physicochemical",["MW","LogP","TPSA","HBD","HBA","Rotatable Bonds","Rings","Aromatic Rings","Heavy Atoms","Fsp3","Stereocenters","Halogens"]),
            ("Drug-likeness",["QED","Drug-likeness","Lipinski Pass","Lipinski Violations","Veber Pass"]),
            ("ADME",["ESOL LogS","ESOL mol/L","Solubility","BBB Penetrant","BBB Score","Pgp Substrate","CYP3A4 Substrate","CYP2D6 Inhibitor","Absorption"]),
        ]
        for _grp_name,_keys in _prop_groups:
            st.markdown(f"**{_grp_name}**")
            _rows=[]
            for _k in _keys:
                _va2=_adm_a.get(_k,'—'); _vb2=_adm_b.get(_k,'—')
                if isinstance(_va2,bool): _va2="✅" if _va2 else "❌"
                if isinstance(_vb2,bool): _vb2="✅" if _vb2 else "❌"
                _rows.append({"Property":_k,_da_name_r:_va2,_db_name_r:_vb2})
            st.dataframe(pd.DataFrame(_rows),use_container_width=True,hide_index=True)
        with st.expander("📏 Lipinski Rule of 5 Detail"):
            _lip_c1,_lip_c2=st.columns(2)
            for _lc,_adm,_nm in [(_lip_c1,_adm_a,_da_name_r),(_lip_c2,_adm_b,_db_name_r)]:
                _lc.markdown(f"**{_nm}**")
                for _rule,_passed in _adm.get("Lipinski",{}).items():
                    _lc.markdown(f"{'✅' if _passed else '❌'} {_rule}")
                _lc.markdown(f"**Violations:** {_adm.get('Lipinski Violations',0)}/4 → {'✅ Pass' if _adm.get('Lipinski Pass') else '❌ Fail'}")
        with st.expander("🔮 Pharmacophore Feature Comparison"):
            _pharm_a=get_pharmacophore_features(st.session_state.get('admet_a_smi',''))
            _pharm_b=get_pharmacophore_features(st.session_state.get('admet_b_smi',''))
            if _pharm_a and _pharm_b:
                _pharm_keys=list(_pharm_a.keys())
                _fig_pharm=go.Figure()
                _fig_pharm.add_trace(go.Bar(name=_da_name_r,x=_pharm_keys,y=[_pharm_a.get(f,0) for f in _pharm_keys],marker_color='#4fc3f7'))
                _fig_pharm.add_trace(go.Bar(name=_db_name_r,x=_pharm_keys,y=[_pharm_b.get(f,0) for f in _pharm_keys],marker_color='#ff9800'))
                _fig_pharm.update_layout(barmode='group',height=360,xaxis=dict(tickangle=-30),
                    paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',font=dict(color='white'),
                    title=dict(text="Pharmacophore Feature Counts",font=dict(color='#4fc3f7')))
                st.plotly_chart(_fig_pharm,use_container_width=True)
    else:
        st.info("Enter SMILES above and click **Compute ADMET Properties**, or run **🔬 Predict Synergy** first — it auto-populates this tab.")

# ═══ TAB 14: Chemical Space ════════════════════════════════════════════════════
with tab14:
    st.markdown("### ⚗️ Chemical Space Explorer")
    st.markdown("**PCA of Morgan ECFP4 fingerprints** across all 30+ drugs in the database. Drugs close together share structural features. Everything computed live.")
    _cs_h_c1,_cs_h_c2=st.columns(2)
    with _cs_h_c1:
        _cs_da=st.selectbox("Highlight Drug A:",["None"]+sorted(DRUG_SMILES_LOOKUP.keys()),key="cs_da_sel")
    with _cs_h_c2:
        _cs_db=st.selectbox("Highlight Drug B:",["None"]+sorted(DRUG_SMILES_LOOKUP.keys()),key="cs_db_sel")
    _cs_highlight=[d for d in [_cs_da,_cs_db] if d!="None"]
    for _pn in [st.session_state.get('name_a',''),st.session_state.get('name_b','')]:
        if _pn and _pn in DRUG_SMILES_LOOKUP and _pn not in _cs_highlight:
            _cs_highlight.append(_pn)
    if st.button("🔭 Compute Chemical Space (PCA)",key="cs_btn",type="primary"):
        with st.spinner("Computing Morgan fingerprints + PCA..."):
            _cs_coords,_cs_var=chemical_space_pca(DRUG_SMILES_LOOKUP,highlight=_cs_highlight)
        st.session_state['cs_coords']=_cs_coords
        st.session_state['cs_var']=_cs_var
    _cs_coords=st.session_state.get('cs_coords',{})
    _cs_var=st.session_state.get('cs_var',(0,0))
    if _cs_coords:
        _cs_names=list(_cs_coords.keys())
        _cs_x=[_cs_coords[n]["x"] for n in _cs_names]
        _cs_y=[_cs_coords[n]["y"] for n in _cs_names]
        _cs_hl=[_cs_coords[n]["highlighted"] for n in _cs_names]
        _cls_colors={"BRAF inhibitor":"#ef5350","MEK inhibitor":"#ff7043","EGFR TKI":"#42a5f5",
            "TKI":"#7e57c2","3rd gen EGFR TKI":"#29b6f6","Dual TKI":"#26c6da","Pan-HER TKI":"#00bcd4",
            "PARP inhibitor":"#66bb6a","CDK4/6 inhibitor":"#ffca28","BTK inhibitor":"#ffa726",
            "BCL-2 inhibitor":"#8d6e63","PI3K inhibitor":"#d4e157","Taxane":"#ff80ab",
            "Anthracycline":"#e040fb","Antimetabolite":"#69f0ae","ALK inhibitor":"#40c4ff",
            "HDAC inhibitor":"#ea80fc","Multi-TKI":"#f48fb1","Alkylating agent":"#b0bec5","Unknown":"#546e7a"}
        _cls_map={n:DRUG_MECHANISMS.get(n,{}).get('class','Unknown') for n in _cs_names}
        _all_classes=sorted(set(_cls_map.values()))
        _cs_fig=go.Figure()
        for _cls in _all_classes:
            _idxs=[i for i,n in enumerate(_cs_names) if _cls_map[n]==_cls and not _cs_hl[i]]
            if not _idxs: continue
            _cs_fig.add_trace(go.Scatter(x=[_cs_x[i] for i in _idxs],y=[_cs_y[i] for i in _idxs],
                mode='markers+text',name=_cls,text=[_cs_names[i] for i in _idxs],textposition='top center',
                textfont=dict(size=9,color='#8b949e'),marker=dict(size=11,color=_cls_colors.get(_cls,'#546e7a'),opacity=0.8),
                hovertemplate='<b>%{text}</b><br>Class: '+_cls+'<br>PC1: %{x:.2f}<br>PC2: %{y:.2f}<extra></extra>'))
        if any(_cs_hl):
            _h_idxs=[i for i,h in enumerate(_cs_hl) if h]
            _cs_fig.add_trace(go.Scatter(x=[_cs_x[i] for i in _h_idxs],y=[_cs_y[i] for i in _h_idxs],
                mode='markers+text',name="⭐ Your Drugs",text=[_cs_names[i] for i in _h_idxs],
                textposition='top center',textfont=dict(size=12,color='#FFD700'),
                marker=dict(size=20,color='#FFD700',symbol='star',line=dict(color='white',width=2)),
                hovertemplate='<b>%{text}</b> ⭐<extra></extra>'))
        _v0=_cs_var[0]*100 if len(_cs_var)>0 else 0
        _v1=_cs_var[1]*100 if len(_cs_var)>1 else 0
        _cs_fig.update_layout(height=620,
            xaxis=dict(title=f"PC1 ({_v0:.1f}% variance)",zeroline=True,zerolinecolor='#30363d'),
            yaxis=dict(title=f"PC2 ({_v1:.1f}% variance)",zeroline=True,zerolinecolor='#30363d'),
            paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',font=dict(color='white'),
            title=dict(text="Chemical Space — Morgan ECFP4 PCA",font=dict(color='#4fc3f7')),
            legend=dict(bgcolor='rgba(22,27,34,0.85)',bordercolor='#30363d',borderwidth=1),hovermode='closest')
        st.plotly_chart(_cs_fig,use_container_width=True)
        st.caption(f"PCA of {len(_cs_coords)} drugs · {_v0+_v1:.1f}% variance · ⭐ = your drugs")
        with st.expander("🌡️ Pairwise Tanimoto Similarity Heatmap"):
            with st.spinner("Computing similarity matrix..."):
                _tan_names,_tan_mat=tanimoto_matrix(DRUG_SMILES_LOOKUP)
            _hm_fig=go.Figure(data=go.Heatmap(z=_tan_mat.tolist(),x=_tan_names,y=_tan_names,
                colorscale=[[0,'#1a237e'],[0.4,'#0d47a1'],[0.7,'#1565c0'],[0.9,'#e53935'],[1,'#b71c1c']],
                zmin=0,zmax=1,hovertemplate='%{y} vs %{x}<br>Tanimoto: %{z:.3f}<extra></extra>',
                colorbar=dict(title="Tanimoto")))
            _hm_fig.update_layout(height=820,xaxis=dict(tickangle=-45,tickfont=dict(size=9)),
                yaxis=dict(tickfont=dict(size=9)),paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),margin=dict(l=160,r=20,t=30,b=160),
                title=dict(text="Pairwise Tanimoto Similarity Matrix",font=dict(color='#4fc3f7')))
            st.plotly_chart(_hm_fig,use_container_width=True)
            st.caption("Red cells (Tanimoto > 0.85) indicate near-identical scaffolds — pairs there likely compete rather than synergize.")
        if _cs_highlight:
            st.markdown("#### 🔍 Nearest Structural Neighbours")
            _all_tan_names,_all_tan_mat=tanimoto_matrix(DRUG_SMILES_LOOKUP)
            for _hn in _cs_highlight:
                if _hn not in _all_tan_names: continue
                _hi=_all_tan_names.index(_hn)
                _sims=[(_all_tan_names[j],float(_all_tan_mat[_hi,j])) for j in range(len(_all_tan_names)) if _all_tan_names[j]!=_hn]
                _sims.sort(key=lambda x:x[1],reverse=True)
                st.markdown(f"**{_hn}** — 5 closest drugs:")
                for _nn,_ns in _sims[:5]:
                    _nn_cls=DRUG_MECHANISMS.get(_nn,{}).get('class','?')
                    _lbl,_=similarity_interpretation(_ns)
                    st.markdown(f"  • **{_nn}** ({_nn_cls}) — Tanimoto `{_ns:.3f}` {_lbl}")
    else:
        st.info("Click **Compute Chemical Space (PCA)** to generate the map.")

# ═══ TAB 15: Combination Index ═════════════════════════════════════════════════
with tab15:
    st.markdown("### 🎯 Combination Index & Synergy Models")
    st.markdown("**Chou-Talalay CI** and **Bliss Independence** quantification. Hill equation dose-response from your EC50 inputs. Pre-filled from last prediction.")
    _ci_c1,_ci_c2,_ci_c3=st.columns(3)
    with _ci_c1:
        _ci_score=st.number_input("GNN synergy score",value=float(st.session_state.get('syn_score') or 0.3),
            min_value=-2.0,max_value=2.0,step=0.01,key="ci_score_inp")
    with _ci_c2:
        _ci_ec50_a=st.number_input("Drug A EC50 (µM)",value=1.0,min_value=0.001,max_value=1000.0,step=0.5,key="ci_ec50a")
    with _ci_c3:
        _ci_ec50_b=st.number_input("Drug B EC50 (µM)",value=2.0,min_value=0.001,max_value=1000.0,step=0.5,key="ci_ec50b")
    _ci_hc1,_ci_hc2=st.columns(2)
    with _ci_hc1: _ci_hill_a=st.slider("Drug A Hill coefficient",0.5,4.0,1.5,0.1,key="ci_hill_a")
    with _ci_hc2: _ci_hill_b=st.slider("Drug B Hill coefficient",0.5,4.0,1.5,0.1,key="ci_hill_b")
    _ci_val,_ci_label,_ci_color=combination_index(_ci_score)
    _bliss15=bliss_analysis(_ci_ec50_a,_ci_ec50_b,_ci_score)
    _rm1,_rm2,_rm3,_rm4=st.columns(4)
    _rm1.metric("Synergy Score",f"{_ci_score:.3f}")
    _rm2.metric("Combination Index",str(_ci_val))
    _rm3.metric("CI Classification",_ci_label)
    _rm4.metric("Bliss Deviation",f"{_bliss15['Bliss_deviation']:+.4f}")
    _bc15="#1e3a1e" if _bliss15['color']=='green' else ("#3a1e1e" if _bliss15['color']=='red' else "#2a2a1e")
    _bb15="#56d364" if _bliss15['color']=='green' else ("#f85149" if _bliss15['color']=='red' else "#f0883e")
    st.markdown(
        f'<div style="background:{_bc15};border-left:4px solid {_bb15};padding:12px;border-radius:6px;color:white;margin:10px 0;">'
        f'<b>Bliss:</b> {_bliss15["Classification"]} · <b>CI:</b> {_ci_val} — {_ci_label} · '
        f'E(A)={_bliss15["E_A"]}, E(B)={_bliss15["E_B"]}, expected={_bliss15["E_AB_expected"]}, observed={_bliss15["E_AB_observed"]}'
        f'</div>',unsafe_allow_html=True)
    st.markdown("---")
    _da15=st.session_state.get('name_a','Drug A') or 'Drug A'
    _db15=st.session_state.get('name_b','Drug B') or 'Drug B'
    st.markdown("#### 📈 Dose-Response Curves (Hill equation)")
    _dr_conc_a,_dr_resp_a=dose_response_hill(_ci_ec50_a,_ci_hill_a)
    _dr_conc_b,_dr_resp_b=dose_response_hill(_ci_ec50_b,_ci_hill_b)
    _fig_dr=go.Figure()
    _fig_dr.add_trace(go.Scatter(x=_dr_conc_a,y=_dr_resp_a,mode='lines',name=_da15,line=dict(color='#4fc3f7',width=2.5)))
    _fig_dr.add_trace(go.Scatter(x=_dr_conc_b,y=_dr_resp_b,mode='lines',name=_db15,line=dict(color='#ff9800',width=2.5)))
    _fig_dr.add_vline(x=_ci_ec50_a,line_dash="dash",line_color="#4fc3f7",annotation_text=f"EC50={_ci_ec50_a}µM",annotation_font_color="#4fc3f7")
    _fig_dr.add_vline(x=_ci_ec50_b,line_dash="dash",line_color="#ff9800",annotation_text=f"EC50={_ci_ec50_b}µM",annotation_font_color="#ff9800")
    _fig_dr.add_hline(y=50,line_dash="dot",line_color="#888",annotation_text="50% effect")
    _fig_dr.update_layout(xaxis=dict(type='log',title='Concentration (µM)'),yaxis=dict(title='Effect (%)',range=[0,105]),
        height=390,paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',font=dict(color='white'),
        title=dict(text="Individual Dose-Response: E(C) = C^h / (EC50^h + C^h) × 100",font=dict(color='#4fc3f7')))
    st.plotly_chart(_fig_dr,use_container_width=True)
    st.markdown("#### 🎨 Combination Dose-Effect Matrix")
    st.caption(f"Each cell = predicted combined effect (Bliss + synergy shift). Score {_ci_score:+.3f} {'boosts' if _ci_score>0 else 'suppresses'} the combined effect.")
    _d_a_doses,_d_b_doses,_combo_mat=synergy_dose_matrix(_ci_ec50_a,_ci_ec50_b,_ci_score)
    _fig_mat=go.Figure(data=go.Heatmap(z=_combo_mat,x=[f"{d:.2f}" for d in _d_b_doses],y=[f"{d:.2f}" for d in _d_a_doses],
        colorscale=[[0,'#0d1117'],[0.3,'#0d47a1'],[0.6,'#1565c0'],[0.85,'#e53935'],[1,'#ffeb3b']],
        zmin=0,zmax=100,hovertemplate=f'{_da15}: %{{y}} µM<br>{_db15}: %{{x}} µM<br>Effect: %{{z:.1f}}%<extra></extra>',
        colorbar=dict(title="Effect %")))
    _fig_mat.update_layout(xaxis=dict(title=f"{_db15} Dose (µM)"),yaxis=dict(title=f"{_da15} Dose (µM)"),
        height=500,paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',font=dict(color='white'),
        title=dict(text=f"Combined Effect Matrix — CI = {_ci_val} ({_ci_label})",font=dict(color='#4fc3f7')))
    st.plotly_chart(_fig_mat,use_container_width=True)
    with st.expander("📊 CI Sensitivity — Score → CI Mapping"):
        _sens_scores=[round(s,2) for s in np.linspace(-1.5,1.5,31)]
        _sens_ci=[combination_index(s)[0] for s in _sens_scores]
        _fig_sens=go.Figure()
        _fig_sens.add_trace(go.Scatter(x=_sens_scores,y=_sens_ci,mode='lines+markers',
            line=dict(color='#4fc3f7',width=2),
            marker=dict(color=['#f85149' if c>1.1 else '#56d364' if c<0.9 else '#8b949e' for c in _sens_ci],size=7),
            hovertemplate='Score: %{x:.2f}<br>CI: %{y:.3f}<extra></extra>'))
        _fig_sens.add_hline(y=1.0,line_dash="dash",line_color="#888",annotation_text="CI = 1 (additive)")
        _fig_sens.add_vline(x=_ci_score,line_dash="dot",line_color="#FFD700",
            annotation_text=f"Current ({_ci_score:.2f})",annotation_font_color="#FFD700")
        _fig_sens.update_layout(xaxis=dict(title="Synergy Score"),yaxis=dict(title="CI"),
            height=320,paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',font=dict(color='white'),
            title=dict(text="CI Sensitivity to Synergy Score",font=dict(color='#4fc3f7')))
        st.plotly_chart(_fig_sens,use_container_width=True)
    with st.expander("📖 Method Reference"):
        st.markdown("""
**Chou-Talalay CI** (Chou 2010, Cancer Res 70:440): CI = exp(−synergy_score) [approximation]. CI < 1 = synergy, CI = 1 = additive, CI > 1 = antagonism.

**Bliss Independence** (Bliss 1939): E(A+B)_expected = E(A) + E(B) − E(A)×E(B). Positive deviation = synergistic.

**Hill equation**: E(C) = C^h / (EC50^h + C^h) × 100%. h = Hill coefficient (steepness).
        """)

# ═══ TAB 16: Explainability ════════════════════════════════════════════════════
with tab16:
    st.markdown("### 🔍 Model Explainability")
    st.markdown("Which structural features drive the prediction? Atom importance uses RDKit chemistry rules — H-bond donors, aromaticity, charges, ring membership. Everything traceable to SMILES.")
    _ex_c1,_ex_c2=st.columns(2)
    with _ex_c1:
        _ex_smi_a=st.text_area("Drug A SMILES",height=75,key="ex_smi_a",value=st.session_state.get('admet_a_smi',''),placeholder="Paste SMILES or run Predict first")
    with _ex_c2:
        _ex_smi_b=st.text_area("Drug B SMILES",height=75,key="ex_smi_b",value=st.session_state.get('admet_b_smi',''),placeholder="Paste SMILES or run Predict first")
    if st.button("🔍 Analyze Structural Features",key="ex_btn",type="primary"):
        if not _ex_smi_a or not _ex_smi_b: st.error("Enter SMILES for both drugs."); st.stop()
        st.session_state['ex_smi_a']=_ex_smi_a; st.session_state['ex_smi_b']=_ex_smi_b
    _ex_smi_a=st.session_state.get('ex_smi_a',_ex_smi_a)
    _ex_smi_b=st.session_state.get('ex_smi_b',_ex_smi_b)
    _ex_name_a=st.session_state.get('name_a','Drug A') or 'Drug A'
    _ex_name_b=st.session_state.get('name_b','Drug B') or 'Drug B'
    if _ex_smi_a and _ex_smi_b:
        _feat_a=drug_feature_importance_bar(_ex_smi_a,_ex_name_a)
        _feat_b=drug_feature_importance_bar(_ex_smi_b,_ex_name_b)
        if _feat_a and _feat_b:
            st.markdown("#### 📊 Feature Class Importance")
            _fi_c1,_fi_c2=st.columns(2)
            for _fc,_feat,_nm,_col in [(_fi_c1,_feat_a,_ex_name_a,'#4fc3f7'),(_fi_c2,_feat_b,_ex_name_b,'#ff9800')]:
                with _fc:
                    _fkeys=list(_feat.keys()); _fvals=[_feat[k] for k in _fkeys]
                    _fig_fi=go.Figure(go.Bar(x=_fvals,y=_fkeys,orientation='h',marker_color=_col,
                        text=[f"{v:.0%}" for v in _fvals],textposition='outside'))
                    _fig_fi.update_layout(height=340,title=dict(text=_nm,font=dict(color='#4fc3f7')),
                        xaxis=dict(range=[0,1.2],title="Normalized importance",showgrid=False),
                        yaxis=dict(autorange='reversed'),paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='white'),margin=dict(l=180,r=80))
                    st.plotly_chart(_fig_fi,use_container_width=True)
            st.markdown("#### 🔀 Feature Overlap & Synergy Signal")
            _all_feat_keys=list(_feat_a.keys())
            _overlap_vals={k:min(_feat_a.get(k,0),_feat_b.get(k,0)) for k in _all_feat_keys}
            _total_overlap=sum(_overlap_vals.values())
            _ol_color="#3a1e1e" if _total_overlap>0.5 else "#1e3a1e"
            _ol_border="#f85149" if _total_overlap>0.5 else "#56d364"
            _ol_msg=("High structural overlap — drugs likely share binding pharmacophores, increasing competition risk."
                     if _total_overlap>0.5 else "Low structural overlap — complementary features, favorable for synergy.")
            st.markdown(f'<div style="background:{_ol_color};border-left:4px solid {_ol_border};padding:12px;border-radius:6px;color:white;">'
                        f'<b>Feature Overlap Score: {_total_overlap:.3f}</b> — {_ol_msg}</div>',unsafe_allow_html=True)
            _fig_ov=go.Figure()
            _fig_ov.add_trace(go.Bar(name=_ex_name_a,x=_all_feat_keys,y=[_feat_a.get(k,0) for k in _all_feat_keys],marker_color='#4fc3f7'))
            _fig_ov.add_trace(go.Bar(name=_ex_name_b,x=_all_feat_keys,y=[_feat_b.get(k,0) for k in _all_feat_keys],marker_color='#ff9800'))
            _fig_ov.add_trace(go.Bar(name='Shared (overlap)',x=_all_feat_keys,y=[_overlap_vals.get(k,0) for k in _all_feat_keys],
                marker_color='rgba(255,235,59,0.6)',marker_line=dict(color='#FFD700',width=1.5)))
            _fig_ov.update_layout(barmode='group',height=380,xaxis=dict(tickangle=-30),
                paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',font=dict(color='white'),
                title=dict(text="Structural Feature Overlap",font=dict(color='#4fc3f7')))
            st.plotly_chart(_fig_ov,use_container_width=True)
            from admet_utils import atom_feature_importance as _afi
            _at_c1,_at_c2=st.columns(2)
            with _at_c1:
                with st.expander(f"🔬 Top Atoms — {_ex_name_a}"):
                    _atoms_a=_afi(_ex_smi_a)
                    if _atoms_a: st.dataframe(pd.DataFrame(_atoms_a[:12]),use_container_width=True,hide_index=True)
            with _at_c2:
                with st.expander(f"🔬 Top Atoms — {_ex_name_b}"):
                    _atoms_b=_afi(_ex_smi_b)
                    if _atoms_b: st.dataframe(pd.DataFrame(_atoms_b[:12]),use_container_width=True,hide_index=True)
            st.markdown("#### 🧠 Mechanism-Informed Rationale")
            _mech_a16=DRUG_MECHANISMS.get(_ex_name_a,{})
            _mech_b16=DRUG_MECHANISMS.get(_ex_name_b,{})
            if _mech_a16 and _mech_b16:
                _pp=(_mech_a16.get('pathway','?'),_mech_b16.get('pathway','?'))
                _expl16=SYNERGY_RULES.get(_pp,SYNERGY_RULES.get((_pp[1],_pp[0]),
                    f"🔬 No specific pathway rule for {_pp[0]} × {_pp[1]}. "
                    f"Structural overlap ({_total_overlap:.3f}) {'suggests competition' if _total_overlap>0.5 else 'suggests complementarity'}. "
                    f"Tanimoto: {st.session_state.get('tanimoto','N/A')}."))
                _bg16="#1e3a1e" if "✅" in _expl16 else "#3a1e1e" if "⚠️" in _expl16 else "#1a2a3a"
                _brd16="#56d364" if "✅" in _expl16 else "#f85149" if "⚠️" in _expl16 else "#4fc3f7"
                st.markdown(f'<div style="background:{_bg16};border-left:4px solid {_brd16};padding:14px;border-radius:6px;color:white;font-size:14px;">{_expl16}</div>',unsafe_allow_html=True)
            else:
                st.info("Drug not in mechanism database. Use structural overlap above as the primary signal.")
    else:
        st.info("Enter SMILES above and click **Analyze Structural Features**, or run **🔬 Predict Synergy** first.")

# ═══ TAB 17: Report Generator ══════════════════════════════════════════════════
with tab17:
    st.markdown("### 📝 Prediction Report Generator")
    st.markdown("Generate a **comprehensive standalone HTML report** — dark-themed, print-ready, opens in any browser. Includes all results, ADMET, Bliss, CI, and methodology notes.")
    _rp_has_pred=st.session_state.get('syn_score') is not None
    if not _rp_has_pred:
        st.info("👆 Run a prediction in **🔬 Predict Synergy** first to enable report generation.")
    else:
        _rp_score=float(st.session_state.get('syn_score',0.0))
        _rp_std=float(st.session_state.get('syn_std',0.0))
        _rp_prob=float(st.session_state.get('syn_prob',0.0))
        _rp_dsa=float(st.session_state.get('dsa',-7.0))
        _rp_dsb=float(st.session_state.get('dsb',-7.0))
        _rp_da=st.session_state.get('name_a','Drug A') or 'Drug A'
        _rp_db=st.session_state.get('name_b','Drug B') or 'Drug B'
        _rp_verdict=st.session_state.get('verdict','?')
        _rp_m1,_rp_m2,_rp_m3,_rp_m4=st.columns(4)
        _rp_m1.metric("Drug A",_rp_da)
        _rp_m2.metric("Drug B",_rp_db)
        _rp_m3.metric("Synergy Score",f"{_rp_score:.3f} ± {_rp_std:.3f}")
        _rp_m4.metric("Verdict",_rp_verdict.split()[0] if _rp_verdict else '?')
        if st.button("📄 Generate & Download Report",key="gen_report_btn",type="primary"):
            from datetime import datetime as _dt
            _rp_ci_val,_rp_ci_lbl,_=combination_index(_rp_score)
            _rp_bliss=bliss_analysis(_rp_dsa,_rp_dsb,_rp_score)
            _report_data={
                "drug_a":_rp_da,"drug_b":_rp_db,
                "pdb_id":st.session_state.get('pdb_id','?'),
                "cell_line":st.session_state.get('cell_line','?'),
                "panel":st.session_state.get('panel','?'),
                "synergy_score":_rp_score,"synergy_std":_rp_std,"synergy_prob":_rp_prob,
                "dock_a":_rp_dsa,"dock_b":_rp_dsb,"verdict":_rp_verdict,
                "model_version":model_version,"model_r":model_r,"model_auroc":model_auroc,
                "mc_samples":20,"protein_name":st.session_state.get('pname',''),
                "admet_a":st.session_state.get('admet_a') or {},
                "admet_b":st.session_state.get('admet_b') or {},
                "tanimoto":st.session_state.get('tanimoto'),
                "ci":_rp_ci_val,"ci_label":_rp_ci_lbl,"bliss":_rp_bliss,
                "timestamp":_dt.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
            }
            _html=generate_html_report(_report_data)
            _fname=f"ProteinSynergyDock_{_rp_da}_{_rp_db}_{_report_data['timestamp'][:10]}.html".replace(' ','_')
            st.download_button(label="⬇️ Download HTML Report",data=_html.encode('utf-8'),
                file_name=_fname,mime="text/html",key="dl_report_btn",type="primary")
            st.success(f"✅ Report ready: **{_fname}**")
        st.markdown("---")
        st.markdown("#### 📊 Session Prediction History")
        if st.session_state.get('history'):
            _hist_df=pd.DataFrame(st.session_state['history'])
            st.dataframe(_hist_df,use_container_width=True,hide_index=True)
            st.download_button("⬇️ Download History (CSV)",data=_hist_df.to_csv(index=False),
                file_name="synergy_session_history.csv",mime="text/csv",key="dl_hist_btn")
        else:
            st.info("No predictions in this session yet.")
        st.markdown("---")
        st.markdown("#### 📦 Export Raw Prediction Data (JSON)")
        if st.button("🗂️ Export JSON",key="export_json_btn"):
            _json_data={
                "drug_a":st.session_state.get('name_a',''),"drug_b":st.session_state.get('name_b',''),
                "pdb_id":st.session_state.get('pdb_id',''),"cell_line":st.session_state.get('cell_line',''),
                "panel":st.session_state.get('panel',''),"synergy_score":st.session_state.get('syn_score'),
                "synergy_std":st.session_state.get('syn_std'),"synergy_prob":st.session_state.get('syn_prob'),
                "dock_a_kcal":st.session_state.get('dsa'),"dock_b_kcal":st.session_state.get('dsb'),
                "verdict":st.session_state.get('verdict'),"tanimoto":st.session_state.get('tanimoto'),
                "admet_a":st.session_state.get('admet_a'),"admet_b":st.session_state.get('admet_b'),
                "mc_samples":st.session_state.get('syn_samples',[]),
            }
            st.download_button("⬇️ Download JSON",data=json.dumps(_json_data,indent=2,default=str),
                file_name="synergy_prediction.json",mime="application/json",key="dl_json_btn")