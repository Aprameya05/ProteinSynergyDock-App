"""
Tests for the docking pipeline's geometry/parsing functions.

These use small synthetic PDB fixtures rather than hitting RCSB over the
network, so the suite runs offline and fast. Network-dependent functions
(fetch_pdb, get_protein_info) are tested separately with mocking.
"""
import os
import tempfile
import pytest
import core

# A minimal synthetic PDB with both protein ATOM records and a HETATM ligand,
# small enough to reason about by hand.
SYNTHETIC_PDB_WITH_LIGAND = """\
ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N
ATOM      2  CA  ALA A   1       1.000   0.000   0.000  1.00  0.00           C
ATOM      3  C   ALA A   1       2.000   0.000   0.000  1.00  0.00           C
HETATM    4  C1  LIG A 101      10.000  10.000  10.000  1.00  0.00           C
HETATM    5  C2  LIG A 101      11.000  10.000  10.000  1.00  0.00           C
HETATM    6  C3  LIG A 101      10.500  11.000  10.000  1.00  0.00           C
HETATM    7  C4  LIG A 101      10.500  10.500  11.000  1.00  0.00           C
HETATM    8  C5  LIG A 101      10.200  10.200  10.200  1.00  0.00           C
HETATM    9  O   HOH A 201      20.000  20.000  20.000  1.00  0.00           O
END
"""

SYNTHETIC_PDB_NO_LIGAND = """\
ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N
ATOM      2  CA  ALA A   1       1.000   0.000   0.000  1.00  0.00           C
ATOM      3  C   ALA A   1       2.000   0.000   0.000  1.00  0.00           C
ATOM      4  N   GLY A   2       3.000   1.000   0.000  1.00  0.00           N
ATOM      5  CA  GLY A   2       4.000   1.000   0.000  1.00  0.00           C
END
"""

SYNTHETIC_PDB_EMPTY = """\
REMARK no coordinates here
END
"""


@pytest.fixture
def pdb_with_ligand_path():
    with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as f:
        f.write(SYNTHETIC_PDB_WITH_LIGAND)
        path = f.name
    yield path
    os.unlink(path)


@pytest.fixture
def pdb_no_ligand_path():
    with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as f:
        f.write(SYNTHETIC_PDB_NO_LIGAND)
        path = f.name
    yield path
    os.unlink(path)


@pytest.fixture
def pdb_empty_path():
    with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as f:
        f.write(SYNTHETIC_PDB_EMPTY)
        path = f.name
    yield path
    os.unlink(path)


class TestGetBindingBox:

    def test_uses_ligand_center_when_hetatm_present(self, pdb_with_ligand_path):
        center, size, method = core.get_binding_box(pdb_with_ligand_path)
        assert method == "ligand"
        # Ligand atoms cluster around (10,10,10) — water (HOH) must be excluded
        assert 9 < center[0] < 11
        assert 9 < center[1] < 11
        assert 9 < center[2] < 11

    def test_excludes_water_from_ligand_centroid(self, pdb_with_ligand_path):
        """The HOH HETATM at (20,20,20) would massively skew the centroid
        if not filtered — this is the actual bug class get_binding_box
        guards against."""
        center, size, method = core.get_binding_box(pdb_with_ligand_path)
        # If water were included, center would be dragged toward 20s.
        assert center[0] < 15

    def test_falls_back_to_protein_center_with_no_ligand(self, pdb_no_ligand_path):
        center, size, method = core.get_binding_box(pdb_no_ligand_path)
        assert method == "protein_center"

    def test_falls_back_to_default_with_no_coordinates(self, pdb_empty_path):
        center, size, method = core.get_binding_box(pdb_empty_path)
        assert method == "default"
        assert center == [0, 0, 0]

    def test_box_size_is_clamped_to_reasonable_range(self, pdb_with_ligand_path):
        """Box sizes should never be degenerate (too small to dock into)
        or absurd (too large for Vina to search efficiently)."""
        _, size, _ = core.get_binding_box(pdb_with_ligand_path)
        for dim in size:
            assert 15 <= dim <= 35, f"box dimension {dim} outside sane docking range"


class TestPoseBlock:
    """pose_block() converts parsed atom tuples back into a minimal PDB
    block for 3D rendering — must round-trip correctly."""

    def test_produces_valid_model_endmdl_wrapper(self):
        atoms = [("C1", 1.0, 2.0, 3.0), ("C2", 4.0, 5.0, 6.0)]
        block = core.pose_block(atoms, chain='A')
        assert block.startswith("MODEL 1")
        assert block.strip().endswith("ENDMDL")
        assert "HETATM" in block

    def test_coordinates_appear_in_output(self):
        atoms = [("C1", 1.234, 2.345, 3.456)]
        block = core.pose_block(atoms)
        assert "1.234" in block
        assert "2.345" in block
        assert "3.456" in block

    def test_empty_atom_list_still_produces_wrapper(self):
        block = core.pose_block([])
        assert "MODEL 1" in block
        assert "ENDMDL" in block


class TestReadPose:

    def test_reads_atoms_from_valid_pdbqt(self, tmp_path):
        pdbqt_content = (
            "MODEL 1\n"
            "ATOM      1  C1  LIG A   1      1.000   2.000   3.000  1.00  0.00     0.000 C\n"
            "ATOM      2  C2  LIG A   1      4.000   5.000   6.000  1.00  0.00     0.000 C\n"
            "ENDMDL\n"
        )
        p = tmp_path / "test.pdbqt"
        p.write_text(pdbqt_content)
        atoms = core.read_pose(str(p))
        assert atoms is not None
        assert len(atoms) == 2

    def test_returns_none_for_missing_file(self):
        assert core.read_pose("/nonexistent/path/file.pdbqt") is None

    def test_returns_none_for_empty_model(self, tmp_path):
        p = tmp_path / "empty.pdbqt"
        p.write_text("MODEL 1\nENDMDL\n")
        assert core.read_pose(str(p)) is None

    def test_stops_at_first_endmdl(self, tmp_path):
        """Only the first pose (binding mode) should be read, not subsequent
        Vina-reported alternative poses in the same file."""
        pdbqt_content = (
            "MODEL 1\n"
            "ATOM      1  C1  LIG A   1      1.000   2.000   3.000  1.00  0.00     0.000 C\n"
            "ENDMDL\n"
            "MODEL 2\n"
            "ATOM      2  C2  LIG A   1      9.000   9.000   9.000  1.00  0.00     0.000 C\n"
            "ENDMDL\n"
        )
        p = tmp_path / "multi.pdbqt"
        p.write_text(pdbqt_content)
        atoms = core.read_pose(str(p))
        assert len(atoms) == 1
        assert atoms[0][1] == 1.000
