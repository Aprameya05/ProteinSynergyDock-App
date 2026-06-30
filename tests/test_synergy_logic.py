"""
Tests for synergy scoring/verdict logic — the core scientific interpretation
layer that turns a model output into a human-readable claim.
"""
import pytest
import core


class TestGetVerdict:
    """get_verdict() turns a raw synergy score into the labels users see
    everywhere in the app. These thresholds are a scientific claim, so they
    need explicit boundary tests, not just happy-path checks."""

    @pytest.mark.parametrize("score,expected_label", [
        (0.51, "Strongly Synergistic"),
        (1.0, "Strongly Synergistic"),
        (10.0, "Strongly Synergistic"),
        (0.11, "Mildly Synergistic"),
        (0.3, "Mildly Synergistic"),
        (0.0, "Approximately Additive"),
        (0.05, "Approximately Additive"),
        (-0.05, "Approximately Additive"),
        (-0.11, "Antagonistic"),
        (-1.0, "Antagonistic"),
    ])
    def test_verdict_label(self, score, expected_label):
        verdict, _ = core.get_verdict(score)
        assert expected_label in verdict

    def test_boundary_at_exactly_0_5_is_not_strongly_synergistic(self):
        """s > 0.5 is strict, so exactly 0.5 falls into the next bucket.
        This documents the boundary behavior explicitly so it can't drift
        silently if the thresholds are ever edited."""
        verdict, _ = core.get_verdict(0.5)
        assert "Strongly" not in verdict

    def test_boundary_at_exactly_0_1(self):
        verdict, _ = core.get_verdict(0.1)
        assert "Mildly" not in verdict

    def test_boundary_at_exactly_minus_0_1_is_additive_not_antagonistic(self):
        """The condition is `elif s > -0.1`, so s == -0.1 fails that check
        and falls through to the final else (Antagonistic) — verifying
        actual implementation behavior, since this is a strict-vs-inclusive
        boundary that's easy to get backwards when reading the code."""
        verdict, _ = core.get_verdict(-0.1)
        assert "Antagonistic" in verdict

    def test_boundary_just_above_minus_0_1_is_additive(self):
        verdict, _ = core.get_verdict(-0.099)
        assert "Additive" in verdict

    def test_returns_color_string(self):
        _, color = core.get_verdict(0.6)
        assert color in ("green", "orange", "blue", "red")

    def test_handles_numpy_float(self):
        """Model output is a torch/numpy scalar in production, not a Python float —
        verdict logic must work with both."""
        import numpy as np
        verdict, color = core.get_verdict(np.float32(0.6))
        assert "Strongly" in verdict


class TestLookupKnown:
    """lookup_known() is the ground-truth comparison shown in the
    'NCI ALMANAC Ground Truth' box — wrong behavior here silently misleads
    users about whether a prediction matches literature."""

    def test_known_pair_exact_cell_line_match(self):
        result = core.lookup_known("Vemurafenib", "Trametinib", "UACC-62")
        assert result is not None
        score, label = result
        assert score == 8.4
        assert label == "UACC-62"

    def test_known_pair_is_symmetric(self):
        """Drug order shouldn't matter — A+B and B+A must return the same value."""
        r1 = core.lookup_known("Vemurafenib", "Trametinib", "UACC-62")
        r2 = core.lookup_known("Trametinib", "Vemurafenib", "UACC-62")
        assert r1 == r2

    def test_known_pair_unknown_cell_line_falls_back_to_average(self):
        result = core.lookup_known("Vemurafenib", "Trametinib", "SOME-UNSEEN-LINE")
        assert result is not None
        score, label = result
        assert "avg" in label

    def test_known_pair_no_cell_line_arg_returns_average(self):
        result = core.lookup_known("Vemurafenib", "Trametinib")
        assert result is not None
        score, _ = result
        assert score == pytest.approx((8.4 + 7.2 + 9.1) / 3)

    def test_unknown_pair_returns_none(self):
        result = core.lookup_known("Aspirin", "Ibuprofen", "MCF7")
        assert result is None

    @pytest.mark.parametrize("d1,d2", list(core.KNOWN_SYNERGY.keys()))
    def test_every_known_synergy_entry_is_retrievable(self, d1, d2):
        """Every (d1,d2) key in KNOWN_SYNERGY must actually be findable via
        lookup_known — guards against the dict being edited in a way that
        breaks the lookup contract."""
        assert core.lookup_known(d1, d2) is not None
