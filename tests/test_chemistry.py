"""
Tests for chemistry/molecular logic: SMILES validity, graph construction.

These tests exist specifically because two invalid SMILES strings
(Rucaparib, Alectinib) shipped to production and only surfaced when a
user hit "Invalid SMILES" at runtime. This file makes that class of bug
fail CI instead of failing in front of a user.
"""
import pytest
from rdkit import Chem
import core


class TestDrugSmilesLookupValidity:
    """Every drug in DRUG_SMILES_LOOKUP must be a chemically valid molecule.
    This is the regression test for the Rucaparib/Alectinib incident."""

    @pytest.mark.parametrize("drug_name", list(core.DRUG_SMILES_LOOKUP.keys()))
    def test_smiles_parses(self, drug_name):
        smiles = core.DRUG_SMILES_LOOKUP[drug_name]
        mol = Chem.MolFromSmiles(smiles)
        assert mol is not None, (
            f"{drug_name} has an invalid SMILES string: {smiles!r}. "
            f"This will surface to users as 'Invalid SMILES for Drug A/B'."
        )

    @pytest.mark.parametrize("drug_name", list(core.DRUG_SMILES_LOOKUP.keys()))
    def test_smiles_has_atoms(self, drug_name):
        """Catches degenerate/empty molecules that parse but are useless."""
        smiles = core.DRUG_SMILES_LOOKUP[drug_name]
        mol = Chem.MolFromSmiles(smiles)
        assert mol.GetNumAtoms() > 3, f"{drug_name} parses but has suspiciously few atoms"

    def test_no_duplicate_canonical_structures(self):
        """Two different drug names shouldn't silently map to the same molecule."""
        canon = {}
        for name, smi in core.DRUG_SMILES_LOOKUP.items():
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            c = Chem.MolToSmiles(mol)
            if c in canon:
                pytest.fail(f"{name} and {canon[c]} have identical canonical SMILES")
            canon[c] = name


class TestShowcaseSmilesValidity:
    """SHOWCASES drives the sidebar quick-pick — every entry must dock successfully."""

    @pytest.mark.parametrize("showcase_name", list(core.SHOWCASES.keys()))
    def test_showcase_smiles_valid_or_intentionally_empty(self, showcase_name):
        entry = core.SHOWCASES[showcase_name]
        sa, sb = entry["smiles_a"], entry["smiles_b"]
        if sa == "" and sb == "":
            return  # "Custom input" placeholder — expected empty
        assert Chem.MolFromSmiles(sa) is not None, f"{showcase_name}: smiles_a invalid"
        assert Chem.MolFromSmiles(sb) is not None, f"{showcase_name}: smiles_b invalid"

    @pytest.mark.parametrize("showcase_name", list(core.SHOWCASES.keys()))
    def test_showcase_drug_names_match_lookup(self, showcase_name):
        """If a showcase references a drug name, that drug should also exist
        in DRUG_SMILES_LOOKUP so the dropdown sync logic finds it."""
        entry = core.SHOWCASES[showcase_name]
        for key in ("name_a", "name_b"):
            name = entry.get(key, "")
            if name and name not in core.DRUG_SMILES_LOOKUP:
                pytest.fail(
                    f"{showcase_name}: {key}={name!r} not in DRUG_SMILES_LOOKUP — "
                    f"showcase->dropdown sync will silently fail to pre-select it."
                )

    @pytest.mark.parametrize("showcase_name", list(core.SHOWCASES.keys()))
    def test_showcase_cell_line_in_panel(self, showcase_name):
        """The pre-selected cell_line must actually belong to the pre-selected panel,
        or the cell-line dropdown index lookup will silently fall back to index 0."""
        entry = core.SHOWCASES[showcase_name]
        panel = entry.get("panel")
        cell_line = entry.get("cell_line")
        if panel and cell_line:
            assert cell_line in core.CANCER_PANELS[panel], (
                f"{showcase_name}: cell_line {cell_line!r} not found in panel {panel!r}"
            )


class TestSmilesToGraph:
    """smiles_to_graph is what actually feeds the model — validate its output shape."""

    def test_valid_smiles_produces_graph(self):
        g = core.smiles_to_graph(core.DRUG_SMILES_LOOKUP["Vemurafenib"])
        assert g is not None
        assert g.x.shape[0] > 0, "graph has zero atoms"
        assert g.x.shape[1] == 7, "expected 7 atom features per node"
        assert g.edge_index.shape[0] == 2, "edge_index should be [2, num_edges]"

    def test_invalid_smiles_returns_none(self):
        assert core.smiles_to_graph("not_a_smiles_string!!!") is None

    def test_empty_string_returns_none(self):
        assert core.smiles_to_graph("") is None

    def test_single_atom_smiles_handled(self):
        # Single atoms have no bonds -> no edges -> function should return None,
        # not crash with an empty edge_index tensor.
        result = core.smiles_to_graph("C")
        assert result is None or result.edge_index.shape[1] == 0

    @pytest.mark.parametrize("drug_name", list(core.DRUG_SMILES_LOOKUP.keys()))
    def test_every_lookup_drug_builds_a_graph(self, drug_name):
        """End-to-end: every drug that passes SMILES validation must also
        successfully build a graph, since that's the actual code path used
        at inference time, not just Chem.MolFromSmiles in isolation."""
        smiles = core.DRUG_SMILES_LOOKUP[drug_name]
        g = core.smiles_to_graph(smiles)
        assert g is not None, f"{drug_name}: valid SMILES but graph construction failed"
