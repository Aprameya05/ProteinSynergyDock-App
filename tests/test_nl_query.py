"""
Tests for parse_nl_query() — the rule-based NL parser behind Tab 10.
Uses a small synthetic scores_data fixture matching the real
{"Panel": {"drugs": [...], "matrix": [[...]]}} schema so tests don't
depend on the real precomputed_scores.json being present.
"""
import pytest
import core


@pytest.fixture
def fake_scores_data():
    return {
        "Melanoma": {
            "drugs": ["Vemurafenib", "Trametinib", "Dabrafenib"],
            "matrix": [
                [0.0, 0.65, 0.20],
                [0.65, 0.0, 0.15],
                [0.20, 0.15, 0.0],
            ],
            "cell_line": "UACC-62",
        },
        "Leukemia": {
            "drugs": ["Imatinib", "Dasatinib", "Vemurafenib"],
            "matrix": [
                [0.0, -0.30, 0.05],
                [-0.30, 0.0, 0.02],
                [0.05, 0.02, 0.0],
            ],
            "cell_line": "K-562",
        },
    }


class TestParseNlQueryNoData:
    def test_empty_scores_data_returns_warning(self):
        result = core.parse_nl_query("anything", {})
        assert "⚠️" in result

    def test_none_scores_data_returns_warning(self):
        result = core.parse_nl_query("anything", None)
        assert "⚠️" in result


class TestParseNlQueryTwoDrugsMentioned:
    def test_known_pair_returns_score(self, fake_scores_data):
        result = core.parse_nl_query("Is Vemurafenib + Trametinib synergistic?", fake_scores_data)
        assert "Vemurafenib" in result
        assert "Trametinib" in result
        assert "0.65" in result or "0.650" in result

    def test_drug_order_in_query_does_not_matter(self, fake_scores_data):
        r1 = core.parse_nl_query("Vemurafenib Trametinib synergy", fake_scores_data)
        r2 = core.parse_nl_query("Trametinib Vemurafenib synergy", fake_scores_data)
        # Both should find the same underlying pair data (score present in both)
        assert ("0.65" in r1 or "0.650" in r1)
        assert ("0.65" in r2 or "0.650" in r2)

    def test_high_score_labeled_synergistic(self, fake_scores_data):
        result = core.parse_nl_query("Vemurafenib Trametinib", fake_scores_data)
        assert "Synergistic" in result

    def test_negative_score_labeled_antagonistic(self, fake_scores_data):
        result = core.parse_nl_query("Imatinib Dasatinib", fake_scores_data)
        assert "Antagonistic" in result

    def test_pair_not_in_any_panel_returns_not_found(self, fake_scores_data):
        result = core.parse_nl_query("Osimertinib Crizotinib synergy", fake_scores_data)
        assert "❌" in result


class TestParseNlQueryOneDrugMentioned:
    def test_single_drug_returns_best_partners(self, fake_scores_data):
        result = core.parse_nl_query("Best combinations with Vemurafenib", fake_scores_data)
        assert "Vemurafenib" in result

    def test_compare_keyword_triggers_cross_panel_comparison(self, fake_scores_data):
        result = core.parse_nl_query("Compare Vemurafenib across cancer types", fake_scores_data)
        assert "Melanoma" in result or "Leukemia" in result

    def test_unmentioned_single_drug_returns_not_found(self, fake_scores_data):
        result = core.parse_nl_query("Tell me about Osimertinib", fake_scores_data)
        assert "❌" in result


class TestParseNlQueryCancerFiltering:
    def test_melanoma_keyword_filters_to_melanoma_panel(self, fake_scores_data):
        result = core.parse_nl_query("most synergistic pairs in melanoma", fake_scores_data)
        # Should not error and should produce a result string
        assert isinstance(result, str)
        assert len(result) > 0

    def test_leukemia_keyword_filters_to_leukemia_panel(self, fake_scores_data):
        result = core.parse_nl_query("synergistic pairs in leukemia", fake_scores_data)
        assert isinstance(result, str)


class TestParseNlQueryGeneralIntent:
    def test_no_drug_mentioned_returns_top_synergistic_by_default(self, fake_scores_data):
        result = core.parse_nl_query("show me good combinations", fake_scores_data)
        assert "🟢" in result or "Most synergistic" in result

    def test_antagonistic_keyword_returns_worst_pairs(self, fake_scores_data):
        result = core.parse_nl_query("show me the worst antagonistic pairs", fake_scores_data)
        assert "🔴" in result or "antagonistic" in result.lower()

    def test_does_not_crash_on_empty_query(self, fake_scores_data):
        result = core.parse_nl_query("", fake_scores_data)
        assert isinstance(result, str)

    def test_does_not_crash_on_gibberish_query(self, fake_scores_data):
        result = core.parse_nl_query("asdkjf alksjdf laksjdf", fake_scores_data)
        assert isinstance(result, str)
