import pytest
from core_fhir import (
    build_diagnostic_report,
    build_operation_outcome,
    predict_to_fhir,
    validate_prediction_input,
    FHIRValidationError,
    NCI60_CELL_LINES,
)


VALID_CELL_LINE = "MCF7"


class TestValidation:
    def test_valid_input_passes(self):
        validate_prediction_input("Olaparib", "Rucaparib", VALID_CELL_LINE, 0.42, 0.79)

    def test_missing_drug_a_raises(self):
        with pytest.raises(FHIRValidationError):
            validate_prediction_input("", "Rucaparib", VALID_CELL_LINE, 0.42, 0.79)

    def test_missing_drug_b_raises(self):
        with pytest.raises(FHIRValidationError):
            validate_prediction_input("Olaparib", None, VALID_CELL_LINE, 0.42, 0.79)

    def test_identical_drugs_raises(self):
        with pytest.raises(FHIRValidationError):
            validate_prediction_input("Olaparib", "olaparib", VALID_CELL_LINE, 0.42, 0.79)

    def test_unknown_cell_line_raises(self):
        with pytest.raises(FHIRValidationError) as exc:
            validate_prediction_input("Olaparib", "Rucaparib", "NOT-A-REAL-LINE", 0.42, 0.79)
        assert exc.value.code == "code-invalid"

    def test_synergy_score_out_of_range_raises(self):
        with pytest.raises(FHIRValidationError):
            validate_prediction_input("Olaparib", "Rucaparib", VALID_CELL_LINE, 5.0, 0.79)

    def test_synergy_score_none_raises(self):
        with pytest.raises(FHIRValidationError):
            validate_prediction_input("Olaparib", "Rucaparib", VALID_CELL_LINE, None, 0.79)

    def test_confidence_out_of_range_raises(self):
        with pytest.raises(FHIRValidationError):
            validate_prediction_input("Olaparib", "Rucaparib", VALID_CELL_LINE, 0.42, 1.5)

    def test_confidence_optional(self):
        validate_prediction_input("Olaparib", "Rucaparib", VALID_CELL_LINE, 0.42, None)

    def test_all_60_cell_lines_present(self):
        assert len(NCI60_CELL_LINES) == 60


class TestDiagnosticReport:
    def test_basic_report_shape(self):
        report = build_diagnostic_report("Olaparib", "Rucaparib", VALID_CELL_LINE, 0.42)
        assert report["resourceType"] == "DiagnosticReport"
        assert report["status"] == "final"
        assert "id" in report
        assert "issued" in report

    def test_report_contains_synergy_observation(self):
        report = build_diagnostic_report("Olaparib", "Rucaparib", VALID_CELL_LINE, 0.42)
        codes = [obs["code"]["coding"][0]["code"] for obs in report["contained"]]
        assert "synergy-score" in codes

    def test_report_includes_confidence_when_provided(self):
        report = build_diagnostic_report(
            "Olaparib", "Rucaparib", VALID_CELL_LINE, 0.42, confidence=0.79
        )
        codes = [obs["code"]["coding"][0]["code"] for obs in report["contained"]]
        assert "model-confidence" in codes

    def test_report_omits_confidence_when_absent(self):
        report = build_diagnostic_report("Olaparib", "Rucaparib", VALID_CELL_LINE, 0.42)
        codes = [obs["code"]["coding"][0]["code"] for obs in report["contained"]]
        assert "model-confidence" not in codes

    def test_report_includes_docking_affinity_when_provided(self):
        report = build_diagnostic_report(
            "Olaparib", "Rucaparib", VALID_CELL_LINE, 0.42, docking_affinity=-8.3
        )
        codes = [obs["code"]["coding"][0]["code"] for obs in report["contained"]]
        assert "docking-affinity" in codes

    def test_invalid_input_raises_not_silently_returns(self):
        with pytest.raises(FHIRValidationError):
            build_diagnostic_report("Olaparib", "Olaparib", VALID_CELL_LINE, 0.42)

    def test_result_references_match_contained_ids(self):
        report = build_diagnostic_report(
            "Olaparib", "Rucaparib", VALID_CELL_LINE, 0.42, confidence=0.79
        )
        contained_ids = {f"#{obs['id']}" for obs in report["contained"]}
        result_refs = {r["reference"] for r in report["result"]}
        assert result_refs == contained_ids

    def test_conclusion_includes_research_disclaimer(self):
        report = build_diagnostic_report("Olaparib", "Rucaparib", VALID_CELL_LINE, 0.42)
        assert "research" in report["conclusion"].lower()
        assert "not a clinical diagnostic" in report["conclusion"].lower()


class TestOperationOutcome:
    def test_shape(self):
        err = FHIRValidationError("bad cell line", code="code-invalid")
        outcome = build_operation_outcome(err)
        assert outcome["resourceType"] == "OperationOutcome"
        assert outcome["issue"][0]["severity"] == "error"
        assert outcome["issue"][0]["code"] == "code-invalid"
        assert outcome["issue"][0]["diagnostics"] == "bad cell line"


class TestPredictToFhir:
    def test_success_returns_diagnostic_report(self):
        resource, success = predict_to_fhir("Olaparib", "Rucaparib", VALID_CELL_LINE, 0.42, 0.79)
        assert success is True
        assert resource["resourceType"] == "DiagnosticReport"

    def test_failure_returns_operation_outcome(self):
        resource, success = predict_to_fhir("Olaparib", "Rucaparib", "FAKE-LINE", 0.42, 0.79)
        assert success is False
        assert resource["resourceType"] == "OperationOutcome"

    def test_does_not_raise_on_invalid_input(self):
        # predict_to_fhir is the safe public entrypoint — it should never
        # propagate an exception to the caller (API layer/Streamlit tab).
        try:
            predict_to_fhir(None, None, None, None, None)
        except Exception as e:
            pytest.fail(f"predict_to_fhir raised unexpectedly: {e}")
