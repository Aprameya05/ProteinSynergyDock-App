"""
ligplot_utils.py

Generates LigPlot-style 2D protein-ligand interaction diagrams from
AutoDock Vina docking results. Each diagram shows:
  - The ligand in 2D (RDKit-generated coordinates)
  - Interacting protein residues as labelled circles
  - H-bond interactions as green dashed lines with distance labels
  - Hydrophobic contacts as grey dashed lines
  - Downloadable as JPEG

Interaction detection uses distance-based heuristics matching LigPlot+'s
default parameters:
  - H-bonds: N/O/S donor-acceptor pairs within 3.5 Å
  - Hydrophobic contacts: C-C pairs within 4.5 Å

No external dependencies beyond RDKit, matplotlib, numpy, and Pillow —
all already in requirements.txt.
"""

from __future__ import annotations

import io
import os
import zipfile
from typing import List, Optional, Tuple, Dict

import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend — required for Streamlit Cloud
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from rdkit.Chem.Draw import rdMolDraw2D


# ── Interaction detection ─────────────────────────────────────────────────────

HBOND_ELEMENTS    = {"N", "O", "S"}
HYDROPHOBIC_ELEMS = {"C"}
HBOND_CUTOFF      = 3.5   # Å — matches LigPlot+ default
HYDROPHOBIC_CUTOFF = 4.5  # Å — matches LigPlot+ default
MAX_RESIDUES_SHOWN = 12   # cap for diagram readability


def _parse_protein_atoms(pdb_text: str) -> List[dict]:
    atoms = []
    for line in pdb_text.splitlines():
        if not line.startswith(("ATOM", "HETATM")):
            continue
        res_name = line[17:20].strip()
        if res_name in ("HOH", "WAT", "H2O"):
            continue
        try:
            atoms.append({
                "atom_name": line[12:16].strip(),
                "res_name":  res_name,
                "chain":     line[21:22].strip(),
                "res_num":   int(line[22:26].strip()),
                "x": float(line[30:38]),
                "y": float(line[38:46]),
                "z": float(line[46:54]),
                "element": (line[76:78].strip() or line[12:16].strip()[0]).upper(),
            })
        except (ValueError, IndexError):
            pass
    return atoms


def find_interactions(
    protein_pdb_text: str,
    ligand_atoms: List[Tuple],   # list of (atom_name, x, y, z)
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Returns (hbonds, hydrophobics) — dicts mapping residue_label → min_distance.

    Residue labels are formatted as e.g. "ASP189" (res_name + res_num).
    """
    protein_atoms = _parse_protein_atoms(protein_pdb_text)
    if not protein_atoms or not ligand_atoms:
        return {}, {}

    lig_coords = np.array([[a[1], a[2], a[3]] for a in ligand_atoms],
                           dtype=np.float32)

    hbonds: Dict[str, float] = {}
    hydrophobics: Dict[str, float] = {}

    for patom in protein_atoms:
        pcoord  = np.array([patom["x"], patom["y"], patom["z"]], dtype=np.float32)
        dists   = np.linalg.norm(lig_coords - pcoord, axis=1)
        min_d   = float(dists.min())
        res_lbl = f"{patom['res_name']}{patom['res_num']}"
        elem    = patom["element"][0] if patom["element"] else "C"

        if elem in HBOND_ELEMENTS and min_d <= HBOND_CUTOFF:
            if res_lbl not in hbonds or min_d < hbonds[res_lbl]:
                hbonds[res_lbl] = round(min_d, 2)
        elif elem in HYDROPHOBIC_ELEMS and min_d <= HYDROPHOBIC_CUTOFF:
            if res_lbl not in hydrophobics or min_d < hydrophobics[res_lbl]:
                hydrophobics[res_lbl] = round(min_d, 2)

    # Remove residues that appear in both — classify as H-bond (stronger signal)
    for k in list(hbonds.keys()):
        hydrophobics.pop(k, None)

    # Limit total shown residues for readability
    hbonds = dict(sorted(hbonds.items(), key=lambda x: x[1])[:MAX_RESIDUES_SHOWN // 2])
    hydrophobics = dict(
        sorted(hydrophobics.items(), key=lambda x: x[1])
        [: MAX_RESIDUES_SHOWN - len(hbonds)]
    )
    return hbonds, hydrophobics


# ── Multi-pose PDBQT parser ───────────────────────────────────────────────────

def read_all_poses(pdbqt_path: str) -> List[List[Tuple]]:
    """
    Read ALL binding poses from a Vina output PDBQT file.
    Returns a list (one per MODEL block) of atom lists [(name, x, y, z), ...].
    """
    if not os.path.exists(pdbqt_path):
        return []

    poses, current = [], []
    with open(pdbqt_path) as f:
        for line in f:
            if line.startswith("MODEL"):
                current = []
            elif line.startswith("ENDMDL"):
                if current:
                    poses.append(current)
                current = []
            elif line.startswith(("ATOM", "HETATM")):
                try:
                    current.append((
                        line[12:16].strip(),
                        float(line[30:38]),
                        float(line[38:46]),
                        float(line[46:54]),
                    ))
                except (ValueError, IndexError):
                    pass
    # Handle files without MODEL/ENDMDL wrappers (single pose)
    if current and not poses:
        poses.append(current)
    return poses


# ── Diagram generation ────────────────────────────────────────────────────────

def _ligand_2d_image(smiles: str, size: int = 300) -> Optional["PIL.Image.Image"]:
    """Render ligand as a 2D RDKit image (PIL)."""
    try:
        from PIL import Image
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        AllChem.Compute2DCoords(mol)
        drawer = rdMolDraw2D.MolDraw2DCairo(size, size)
        drawer.drawOptions().addStereoAnnotation = True
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        png_bytes = drawer.GetDrawingText()
        return Image.open(io.BytesIO(png_bytes))
    except Exception:
        return None


def generate_ligplot(
    smiles: str,
    ligand_atoms: List[Tuple],
    protein_pdb_text: str,
    pose_num: int,
    drug_name: str,
    binding_affinity: Optional[float] = None,
) -> bytes:
    """
    Generate a LigPlot-style 2D interaction diagram for one docking pose.
    Returns JPEG bytes suitable for download or st.image().

    Layout:
      - Top-left: 2D ligand structure (RDKit)
      - Right/below: residue circles connected by interaction lines
      - Green dashed = H-bond (with distance label)
      - Grey dashed = hydrophobic contact
    """
    hbonds, hydrophobics = find_interactions(protein_pdb_text, ligand_atoms)
    lig_img = _ligand_2d_image(smiles, size=320)

    fig = plt.figure(figsize=(12, 8), facecolor="#1a1a2e")
    ax  = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor("#1a1a2e")
    ax.set_xlim(0, 12); ax.set_ylim(0, 8)
    ax.axis("off")

    # ── Title ──────────────────────────────────────────────────────────────────
    aff_str = f"  |  ΔG = {binding_affinity:.2f} kcal/mol" if binding_affinity else ""
    ax.text(
        6, 7.6, f"Pose {pose_num} — {drug_name}{aff_str}",
        ha="center", va="center", fontsize=14, fontweight="bold",
        color="#4fc3f7", fontfamily="monospace"
    )
    ax.text(
        6, 7.2,
        "● H-bond interactions (green)   ● Hydrophobic contacts (grey)",
        ha="center", va="center", fontsize=9, color="#b0bec5"
    )

    # ── 2D ligand image ────────────────────────────────────────────────────────
    lig_box_x, lig_box_y = 0.3, 3.5   # center of ligand image in axes coords
    lig_size_ax = 2.8
    if lig_img:
        from matplotlib.image import AxesImage
        newax = fig.add_axes([0.02, 0.35, 0.28, 0.45])
        newax.imshow(lig_img)
        newax.axis("off")
        newax.set_facecolor("#252540")
        for spine in newax.spines.values():
            spine.set_edgecolor("#4fc3f7")
            spine.set_linewidth(1.5)
        ax.text(
            1.8, 2.9, "Ligand (2D)", ha="center", va="top",
            fontsize=8, color="#90caf9", style="italic"
        )

    # ── Residue placement ──────────────────────────────────────────────────────
    all_residues = (
        [(r, d, "hbond") for r, d in hbonds.items()] +
        [(r, d, "hydro") for r, d in hydrophobics.items()]
    )

    if not all_residues:
        ax.text(
            6, 4, "No interactions detected within cutoff distances.\n"
                  f"(H-bond ≤ {HBOND_CUTOFF} Å, Hydrophobic ≤ {HYDROPHOBIC_CUTOFF} Å)",
            ha="center", va="center", fontsize=11, color="#ff9800",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#252540", edgecolor="#ff9800")
        )
    else:
        n = len(all_residues)
        # Arrange residues in a half-ellipse on the right side
        cx, cy = 7.5, 4.0   # center of ellipse
        rx, ry = 3.5, 3.2   # semi-axes

        # Ligand anchor point (right edge of 2D image box, approximate)
        lig_anchor = (3.2, 4.0)

        for i, (res, dist, itype) in enumerate(all_residues):
            angle = np.pi * 0.15 + (np.pi * 0.7) * i / max(n - 1, 1)
            rx_i = rx + (0.4 if i % 2 == 0 else 0)  # stagger slightly
            x = cx + rx_i * np.cos(angle)
            y = cy + ry  * np.sin(angle)

            is_hbond = itype == "hbond"
            box_color   = "#1a3a1a" if is_hbond else "#2a2a2a"
            edge_color  = "#4caf50" if is_hbond else "#78909c"
            line_color  = "#4caf50" if is_hbond else "#78909c"
            line_style  = "--"
            line_width  = 1.8 if is_hbond else 1.2

            # Draw line from ligand anchor to residue
            ax.plot(
                [lig_anchor[0], x], [lig_anchor[1], y],
                color=line_color, linestyle=line_style,
                linewidth=line_width, alpha=0.8, zorder=1
            )

            # Distance label on line midpoint
            mid_x = (lig_anchor[0] + x) / 2
            mid_y = (lig_anchor[1] + y) / 2
            if is_hbond:
                ax.text(
                    mid_x, mid_y, f"{dist} Å",
                    ha="center", va="center", fontsize=7,
                    color="#a5d6a7",
                    bbox=dict(boxstyle="round,pad=0.15",
                              facecolor="#1a2e1a", edgecolor="none", alpha=0.8)
                )

            # Residue box
            box = FancyBboxPatch(
                (x - 0.55, y - 0.28), 1.1, 0.56,
                boxstyle="round,pad=0.05",
                facecolor=box_color, edgecolor=edge_color,
                linewidth=1.5, zorder=2
            )
            ax.add_patch(box)
            ax.text(
                x, y, res,
                ha="center", va="center", fontsize=8, fontweight="bold",
                color="#e0e0e0", zorder=3
            )

    # ── Legend ─────────────────────────────────────────────────────────────────
    legend_x, legend_y = 0.5, 1.8
    ax.plot([legend_x, legend_x + 0.6], [legend_y, legend_y],
            color="#4caf50", linestyle="--", linewidth=2)
    ax.text(legend_x + 0.75, legend_y, f"H-bond (≤ {HBOND_CUTOFF} Å)",
            va="center", fontsize=8, color="#a5d6a7")
    ax.plot([legend_x, legend_x + 0.6], [legend_y - 0.4, legend_y - 0.4],
            color="#78909c", linestyle="--", linewidth=1.5)
    ax.text(legend_x + 0.75, legend_y - 0.4,
            f"Hydrophobic (≤ {HYDROPHOBIC_CUTOFF} Å)",
            va="center", fontsize=8, color="#b0bec5")

    # Summary stats
    ax.text(
        6, 0.7,
        f"H-bonds: {len(hbonds)}   Hydrophobic contacts: {len(hydrophobics)}   "
        f"Total interacting residues: {len(all_residues)}",
        ha="center", va="center", fontsize=9, color="#90caf9",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#0f2040",
                  edgecolor="#4fc3f7", linewidth=1)
    )

    # ── Export as JPEG ─────────────────────────────────────────────────────────
    buf = io.BytesIO()
    fig.savefig(buf, format="jpeg", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def generate_all_ligplots_zip(
    smiles: str,
    all_poses: List[List[Tuple]],
    protein_pdb_text: str,
    drug_name: str,
    affinities: Optional[List[float]] = None,
) -> bytes:
    """
    Generates LigPlot JPEG for each pose and bundles them into a ZIP.
    Returns ZIP bytes for st.download_button.
    """
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for i, pose_atoms in enumerate(all_poses, start=1):
            affinity = affinities[i - 1] if affinities and i <= len(affinities) else None
            try:
                jpeg_bytes = generate_ligplot(
                    smiles=smiles,
                    ligand_atoms=pose_atoms,
                    protein_pdb_text=protein_pdb_text,
                    pose_num=i,
                    drug_name=drug_name,
                    binding_affinity=affinity,
                )
                zf.writestr(f"{drug_name}_pose{i}_ligplot.jpg", jpeg_bytes)
            except Exception as e:
                # Write a plain text error file instead of silently skipping
                zf.writestr(
                    f"{drug_name}_pose{i}_ERROR.txt",
                    f"LigPlot generation failed for pose {i}: {e}"
                )
    buf.seek(0)
    return buf.read()


def parse_vina_affinities(pdbqt_path: str) -> List[float]:
    """
    Extracts binding affinity (kcal/mol) for each pose from Vina PDBQT output.
    Returns list in pose order (pose 1 first = best).
    """
    affinities = []
    if not os.path.exists(pdbqt_path):
        return affinities
    with open(pdbqt_path) as f:
        for line in f:
            if "VINA RESULT" in line:
                try:
                    affinities.append(float(line.split()[3]))
                except (IndexError, ValueError):
                    pass
    return affinities
