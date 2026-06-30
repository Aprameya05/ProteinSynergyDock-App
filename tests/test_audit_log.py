import os
import pytest
from audit_log import AuditLog, GENESIS_HASH


@pytest.fixture
def temp_log(tmp_path):
    path = str(tmp_path / "test_audit.jsonl")
    return AuditLog(path=path)


class TestAuditLog:
    def test_empty_log_last_hash_is_genesis(self, temp_log):
        assert temp_log._last_hash() == GENESIS_HASH

    def test_record_creates_file(self, temp_log):
        temp_log.record(
            "Olaparib", "Rucaparib", "MCF7",
            "DiagnosticReport", "synergy=0.42", "v2", True,
        )
        assert os.path.exists(temp_log.path)

    def test_record_returns_entry_with_required_fields(self, temp_log):
        entry = temp_log.record(
            "Olaparib", "Rucaparib", "MCF7",
            "DiagnosticReport", "synergy=0.42", "v2", True,
        )
        for field in ("timestamp", "input_hash", "entry_hash", "prev_hash", "success"):
            assert field in entry

    def test_first_entry_chains_to_genesis(self, temp_log):
        entry = temp_log.record(
            "Olaparib", "Rucaparib", "MCF7",
            "DiagnosticReport", "synergy=0.42", "v2", True,
        )
        assert entry["prev_hash"] == GENESIS_HASH

    def test_second_entry_chains_to_first(self, temp_log):
        e1 = temp_log.record("A", "B", "MCF7", "DiagnosticReport", "x", "v2", True)
        e2 = temp_log.record("C", "D", "HCT-116", "DiagnosticReport", "y", "v2", True)
        assert e2["prev_hash"] == e1["entry_hash"]

    def test_read_all_returns_all_entries_in_order(self, temp_log):
        temp_log.record("A", "B", "MCF7", "DiagnosticReport", "x", "v2", True)
        temp_log.record("C", "D", "HCT-116", "DiagnosticReport", "y", "v2", True)
        temp_log.record("E", "F", "PC-3", "DiagnosticReport", "z", "v2", True)
        entries = temp_log.read_all()
        assert len(entries) == 3
        assert [e["drug_a"] for e in entries] == ["A", "C", "E"]

    def test_failed_predictions_are_also_logged(self, temp_log):
        temp_log.record("A", "A", "MCF7", "OperationOutcome", "invalid: same drug", "v2", False)
        entries = temp_log.read_all()
        assert entries[0]["success"] is False
        assert entries[0]["output_resource_type"] == "OperationOutcome"

    def test_verify_chain_valid_on_clean_log(self, temp_log):
        for i in range(5):
            temp_log.record(f"Drug{i}", f"Drug{i+1}", "MCF7", "DiagnosticReport", "x", "v2", True)
        valid, broken_at = temp_log.verify_chain()
        assert valid is True
        assert broken_at is None

    def test_verify_chain_detects_tampering(self, temp_log):
        temp_log.record("A", "B", "MCF7", "DiagnosticReport", "x", "v2", True)
        temp_log.record("C", "D", "HCT-116", "DiagnosticReport", "y", "v2", True)

        # Tamper: rewrite the first line's drug_a without recomputing hashes
        import json
        with open(temp_log.path, "r") as f:
            lines = f.readlines()
        first = json.loads(lines[0])
        first["drug_a"] = "TAMPERED"
        lines[0] = json.dumps(first, sort_keys=True) + "\n"
        with open(temp_log.path, "w") as f:
            f.writelines(lines)

        valid, broken_at = temp_log.verify_chain()
        assert valid is False
        assert broken_at == 0

    def test_input_hash_consistent_for_same_inputs(self, temp_log):
        e1 = temp_log.record("A", "B", "MCF7", "DiagnosticReport", "x", "v2", True)
        e2 = temp_log.record("A", "B", "MCF7", "DiagnosticReport", "x2", "v2", True)
        assert e1["input_hash"] == e2["input_hash"]

    def test_user_field_defaults_to_anonymous(self, temp_log):
        entry = temp_log.record("A", "B", "MCF7", "DiagnosticReport", "x", "v2", True)
        assert entry["user"] == "anonymous"

    def test_user_field_can_be_set(self, temp_log):
        entry = temp_log.record(
            "A", "B", "MCF7", "DiagnosticReport", "x", "v2", True, user="researcher_1"
        )
        assert entry["user"] == "researcher_1"
