"""
Tests for MUTATION_DB integrity — Tab 8's resistance analysis depends on
this data being internally consistent (every drug referenced as "affected"
must exist in the drug list shown in the dropdown).
"""
import pytest
import core

# The exact drug list exposed in the Tab 8 "Drug to Test" dropdown in app.py.
# Kept here explicitly (not imported) so this test catches drift between
# the dropdown options and MUTATION_DB's drugs_affected references.
TAB8_DROPDOWN_DRUGS = [
    "Vemurafenib", "Erlotinib", "Imatinib", "Dasatinib", "Crizotinib",
    "Osimertinib", "Gefitinib", "Dabrafenib", "Alectinib", "Nilotinib",
]


class TestMutationDbIntegrity:

    @pytest.mark.parametrize("gene", list(core.MUTATION_DB.keys()))
    def test_gene_has_wild_type_pdb(self, gene):
        assert core.MUTATION_DB[gene]["wild_type"], f"{gene} missing wild_type PDB ID"

    @pytest.mark.parametrize("gene", list(core.MUTATION_DB.keys()))
    def test_gene_has_at_least_one_mutation(self, gene):
        assert len(core.MUTATION_DB[gene]["mutations"]) > 0

    def test_every_drugs_affected_entry_is_in_tab8_dropdown(self):
        """If MUTATION_DB references a drug the Tab 8 selectbox doesn't offer,
        that mutation's resistance profile can never actually be tested by a user."""
        missing = []
        for gene, gdata in core.MUTATION_DB.items():
            for mut_name, mut_info in gdata["mutations"].items():
                for drug in mut_info["drugs_affected"]:
                    if drug not in TAB8_DROPDOWN_DRUGS:
                        missing.append(f"{gene} {mut_name} references {drug!r}")
        assert not missing, "Drugs referenced in MUTATION_DB but absent from Tab 8 dropdown:\n" + "\n".join(missing)

    def test_every_drugs_affected_entry_is_in_smiles_lookup(self):
        """Drugs referenced in resistance data should also be dockable
        (i.e. present in DRUG_SMILES_LOOKUP) for consistency across tabs."""
        missing = []
        for gene, gdata in core.MUTATION_DB.items():
            for mut_name, mut_info in gdata["mutations"].items():
                for drug in mut_info["drugs_affected"]:
                    if drug not in core.DRUG_SMILES_LOOKUP:
                        missing.append(f"{gene} {mut_name}: {drug!r}")
        assert not missing, "Drugs in MUTATION_DB but missing from DRUG_SMILES_LOOKUP:\n" + "\n".join(missing)

    @pytest.mark.parametrize("gene", list(core.MUTATION_DB.keys()))
    def test_mutation_entries_have_required_fields(self, gene):
        for mut_name, mut_info in core.MUTATION_DB[gene]["mutations"].items():
            assert "pdb" in mut_info and mut_info["pdb"], f"{gene} {mut_name} missing pdb"
            assert "description" in mut_info and mut_info["description"], f"{gene} {mut_name} missing description"
            assert "drugs_affected" in mut_info, f"{gene} {mut_name} missing drugs_affected"
            assert len(mut_info["drugs_affected"]) > 0, f"{gene} {mut_name} has empty drugs_affected"

    def test_known_biology_braf_v600e_affects_vemurafenib(self):
        """Spot-check against known pharmacology, not just schema validity —
        BRAF V600E is THE textbook resistance mutation for Vemurafenib."""
        assert "Vemurafenib" in core.MUTATION_DB["BRAF"]["mutations"]["V600E"]["drugs_affected"]

    def test_known_biology_egfr_t790m_affects_first_gen_not_necessarily_osimertinib(self):
        """T790M is the classic gatekeeper resistance mutation for 1st-gen
        EGFR inhibitors (erlotinib/gefitinib). Osimertinib was specifically
        designed to overcome it, so it being listed as 'affected' here would
        be a biologically backwards claim worth flagging if it ever appears."""
        affected = core.MUTATION_DB["EGFR"]["mutations"]["T790M"]["drugs_affected"]
        assert "Erlotinib" in affected
        assert "Gefitinib" in affected
