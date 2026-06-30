"""
Tests for MC Dropout uncertainty quantification (predict_with_uncertainty,
_enable_mc_dropout, confidence_label).

These tests exist to catch the single most dangerous failure mode of MC
Dropout: silently NOT being stochastic (e.g. because eval() was called
after enabling dropout, or because dropout probability is 0, or because
the wrong submodules got toggled). A bug here wouldn't crash anything —
it would just silently report std=0.0 ("High confidence") on every
prediction, which is worse than not having uncertainty at all because
it actively misleads.
"""
import pytest
import torch
import torch.nn as nn
import numpy as np
from torch_geometric.data import Data, Batch
import core


class TinyDockModel(nn.Module):
    """Minimal stand-in for ProteinSynergyDockV1/V2 with the same dropout
    pattern (two dropout layers in the prediction head), so tests don't
    need a real trained checkpoint."""
    def __init__(self, p1=0.5, p2=0.3):
        super().__init__()
        self.drop1 = nn.Dropout(p1)
        self.drop2 = nn.Dropout(p2)
        self.lin = nn.Linear(10, 2)

    def forward(self, da, db, go_emb, dock):
        x = torch.cat([torch.ones(1, 8), dock], dim=-1)
        x = self.drop1(x)
        x = self.drop2(x)
        out = self.lin(x)
        return out[:, 0], out[:, 1]


@pytest.fixture
def tiny_model():
    return TinyDockModel()


@pytest.fixture
def dummy_graph_inputs():
    ga = Data(x=torch.randn(3, 7), edge_index=torch.tensor([[0, 1], [1, 0]]))
    gb = Data(x=torch.randn(3, 7), edge_index=torch.tensor([[0, 1], [1, 0]]))
    go_emb = torch.zeros(512).unsqueeze(0)
    dock = torch.tensor([[-7.5, -8.0]])
    return ga, gb, go_emb, dock


class TestEnableMcDropout:

    def test_only_dropout_layers_set_to_train_mode(self):
        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.lin = nn.Linear(4, 4)
                self.drop = nn.Dropout(0.5)
                self.norm = nn.LayerNorm(4)
        m = M()
        m.eval()
        core._enable_mc_dropout(m)
        assert m.drop.training is True, "Dropout must be in train mode for MC sampling"
        assert m.lin.training is False, "Linear layers must stay in eval mode"
        assert m.norm.training is False, "Norm layers must stay in eval mode"

    def test_multiple_dropout_layers_all_enabled(self, tiny_model):
        tiny_model.eval()
        core._enable_mc_dropout(tiny_model)
        assert tiny_model.drop1.training is True
        assert tiny_model.drop2.training is True

    def test_idempotent_does_not_error_on_repeated_calls(self, tiny_model):
        tiny_model.eval()
        core._enable_mc_dropout(tiny_model)
        core._enable_mc_dropout(tiny_model)  # should not raise
        assert tiny_model.drop1.training is True


class TestPredictWithUncertainty:

    def test_returns_required_keys(self, tiny_model, dummy_graph_inputs):
        ga, gb, go_emb, dock = dummy_graph_inputs
        result = core.predict_with_uncertainty(
            tiny_model, 'v1', None, ga, gb, go_emb, dock, None, Batch, n_samples=10
        )
        for key in ("mean_synergy", "std_synergy", "mean_prob", "std_prob", "synergy_samples", "n_samples"):
            assert key in result

    def test_n_samples_respected(self, tiny_model, dummy_graph_inputs):
        ga, gb, go_emb, dock = dummy_graph_inputs
        result = core.predict_with_uncertainty(
            tiny_model, 'v1', None, ga, gb, go_emb, dock, None, Batch, n_samples=15
        )
        assert len(result["synergy_samples"]) == 15
        assert result["n_samples"] == 15

    def test_dropout_produces_nonzero_variance(self, tiny_model, dummy_graph_inputs):
        """The critical test: if this ever returns std=0.0, MC Dropout has
        silently broken (most likely cause: model.eval() called after
        _enable_mc_dropout instead of before, or dropout p=0)."""
        ga, gb, go_emb, dock = dummy_graph_inputs
        result = core.predict_with_uncertainty(
            tiny_model, 'v1', None, ga, gb, go_emb, dock, None, Batch, n_samples=30
        )
        assert result["std_synergy"] > 0.0, (
            "MC Dropout produced zero variance across 30 samples — "
            "dropout is not actually stochastic at inference time."
        )

    def test_samples_are_not_all_identical(self, tiny_model, dummy_graph_inputs):
        ga, gb, go_emb, dock = dummy_graph_inputs
        result = core.predict_with_uncertainty(
            tiny_model, 'v1', None, ga, gb, go_emb, dock, None, Batch, n_samples=20
        )
        unique_values = set(round(s, 6) for s in result["synergy_samples"])
        assert len(unique_values) > 1, "All MC samples identical — dropout had no effect"

    def test_mean_is_within_min_max_of_samples(self, tiny_model, dummy_graph_inputs):
        ga, gb, go_emb, dock = dummy_graph_inputs
        result = core.predict_with_uncertainty(
            tiny_model, 'v1', None, ga, gb, go_emb, dock, None, Batch, n_samples=20
        )
        samples = result["synergy_samples"]
        assert min(samples) <= result["mean_synergy"] <= max(samples)

    def test_model_restored_to_full_eval_mode_after_call(self, tiny_model, dummy_graph_inputs):
        """After predict_with_uncertainty returns, the model must be back
        in full eval mode — otherwise a subsequent normal (non-MC) call
        elsewhere in the app would unknowingly run with dropout still on."""
        ga, gb, go_emb, dock = dummy_graph_inputs
        core.predict_with_uncertainty(
            tiny_model, 'v1', None, ga, gb, go_emb, dock, None, Batch, n_samples=5
        )
        assert tiny_model.drop1.training is False
        assert tiny_model.drop2.training is False

    def test_more_samples_gives_more_stable_std_estimate(self, tiny_model, dummy_graph_inputs):
        """Not a strict mathematical guarantee for any single run, but
        sanity-checks that n_samples actually changes the sampling
        behavior rather than being ignored."""
        ga, gb, go_emb, dock = dummy_graph_inputs
        torch.manual_seed(42)
        r_small = core.predict_with_uncertainty(
            tiny_model, 'v1', None, ga, gb, go_emb, dock, None, Batch, n_samples=3
        )
        torch.manual_seed(42)
        r_large = core.predict_with_uncertainty(
            tiny_model, 'v1', None, ga, gb, go_emb, dock, None, Batch, n_samples=50
        )
        assert len(r_small["synergy_samples"]) == 3
        assert len(r_large["synergy_samples"]) == 50

    def test_v2_path_with_cell_index(self, dummy_graph_inputs):
        class TinyV2(nn.Module):
            def __init__(self):
                super().__init__()
                self.drop = nn.Dropout(0.5)
                self.lin = nn.Linear(11, 2)
                self.cell_embed = nn.Embedding(5, 1)
            def forward(self, da, db, go_emb, dock, cell_idx):
                x = torch.cat([torch.ones(1, 8), dock, self.cell_embed(cell_idx).squeeze(0).unsqueeze(0)], dim=-1)
                x = self.drop(x)
                out = self.lin(x)
                return out[:, 0], out[:, 1]
        ga, gb, go_emb, dock = dummy_graph_inputs
        m = TinyV2()
        cell_to_idx = {"UACC-62": 0, "MCF7": 1}
        result = core.predict_with_uncertainty(
            m, 'v2', cell_to_idx, ga, gb, go_emb, dock, "UACC-62", Batch, n_samples=10
        )
        assert result["std_synergy"] >= 0.0
        assert len(result["synergy_samples"]) == 10


class TestConfidenceLabel:

    @pytest.mark.parametrize("std,expected_substring", [
        (0.0, "High confidence"),
        (0.1, "High confidence"),
        (0.14, "High confidence"),
        (0.15, "Moderate confidence"),
        (0.3, "Moderate confidence"),
        (0.39, "Moderate confidence"),
        (0.4, "Low confidence"),
        (1.0, "Low confidence"),
        (5.0, "Low confidence"),
    ])
    def test_confidence_band_thresholds(self, std, expected_substring):
        label, _ = core.confidence_label(std)
        assert expected_substring in label

    def test_returns_color_for_ui(self):
        _, color = core.confidence_label(0.05)
        assert color in ("green", "orange", "red")

    def test_handles_numpy_float(self):
        label, _ = core.confidence_label(np.float64(0.2))
        assert "Moderate" in label
