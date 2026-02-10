import pytest
import torch
import numpy as np
import os
import sys

# Add parent directory to Python path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

from gns.learned_simulator import LearnedSimulator


@pytest.fixture
def simulator():
    """Fixture to create a simulator object"""
    particle_dimensions = 2
    nnode_in = 30
    nedge_in = 3
    latent_dim = 128
    nmessage_passing_steps = 5
    nmlp_layers = 2
    mlp_hidden_dim = 64
    connectivity_radius = 0.05
    boundaries = np.array([[-1.0, 1.0], [-1.0, 1.0]])
    normalization_stats = {
        "acceleration": {"mean": 0.0, "std": 1.0},
        "velocity": {"mean": 0.0, "std": 1.0},
    }
    nparticle_types = 1
    particle_type_embedding_size = 16
    boundary_clamp_limit = 1.0
    device = "cpu"

    return LearnedSimulator(
        particle_dimensions,
        nnode_in,
        nedge_in,
        latent_dim,
        nmessage_passing_steps,
        nmlp_layers,
        mlp_hidden_dim,
        connectivity_radius,
        boundaries,
        normalization_stats,
        nparticle_types,
        particle_type_embedding_size,
        boundary_clamp_limit,
        device,
    )


def test_encoder_preprocessor(simulator):
    """Test for _encoder_preprocessor"""
    position_sequence = torch.tensor(
        [
            [[0.0, 0.0], [0.0, 0.1], [0.0, 0.2], [0.0, 0.3], [0.0, 0.4], [0.0, 0.5]],
            [[0.0, 0.0], [0.0, 0.2], [0.0, 0.4], [0.0, 0.6], [0.0, 0.8], [0.0, 1.0]],
        ]
    )
    nparticles_per_example = torch.tensor([2])
    particle_types = torch.tensor([0, 0])

    node_features, edge_index, edge_features = simulator._encoder_preprocessor(
        position_sequence, nparticles_per_example, particle_types
    )

    assert node_features.shape == (2, 14)
    assert edge_index.shape == (2, 2)  # one edge between the 2 particles
    assert edge_features.shape == (2, 3)  # one edge with 3 features

    # check the constructed graph has the expected nodes
    assert set(edge_index[0].tolist()) == set([0, 1])  # senders
    assert set(edge_index[1].tolist()) == set([0, 1])  # receivers

    # check the constructed graph has the expected edge
    assert {tuple(edge_index[:, i].tolist()) for i in range(edge_index.shape[1])} == {
        (0, 0),
        (1, 1),
    }


def test_rigid_body_configuration():
    particle_dimensions = 2
    nnode_in = 30
    nedge_in = 3
    latent_dim = 128
    nmessage_passing_steps = 5
    nmlp_layers = 2
    mlp_hidden_dim = 64
    connectivity_radius = 0.05
    boundaries = np.array([[-1.0, 1.0], [-1.0, 1.0]])
    normalization_stats = {
        "acceleration": {"mean": 0.0, "std": 1.0},
        "velocity": {"mean": 0.0, "std": 1.0},
    }
    nparticle_types = 1
    particle_type_embedding_size = 16
    boundary_clamp_limit = 1.0
    device = "cpu"

    model = LearnedSimulator(
        particle_dimensions,
        nnode_in,
        nedge_in,
        latent_dim,
        nmessage_passing_steps,
        nmlp_layers,
        mlp_hidden_dim,
        connectivity_radius,
        boundaries,
        normalization_stats,
        nparticle_types,
        particle_type_embedding_size,
        boundary_clamp_limit,
        device,
        rigid_bodies=[[0, 1, 2]],
    )

    assert model.rigid_bodies is not None
    assert len(model.rigid_bodies) == 1
    assert torch.equal(model.rigid_bodies[0], torch.tensor([0, 1, 2], dtype=torch.long))

    model.set_rigid_bodies([[1, 3]])
    assert len(model.rigid_bodies) == 1
    assert torch.equal(model.rigid_bodies[0], torch.tensor([1, 3], dtype=torch.long))


def test_forward_passes_current_velocities_to_rigid_constraint(monkeypatch):
    particle_dimensions = 2
    nnode_in = 30
    nedge_in = 3
    latent_dim = 128
    nmessage_passing_steps = 5
    nmlp_layers = 2
    mlp_hidden_dim = 64
    connectivity_radius = 0.05
    boundaries = np.array([[-1.0, 1.0], [-1.0, 1.0]])
    normalization_stats = {
        "acceleration": {"mean": 0.0, "std": 1.0},
        "velocity": {"mean": 0.0, "std": 1.0},
    }
    nparticle_types = 1
    particle_type_embedding_size = 16
    boundary_clamp_limit = 1.0
    device = "cpu"

    model = LearnedSimulator(
        particle_dimensions,
        nnode_in,
        nedge_in,
        latent_dim,
        nmessage_passing_steps,
        nmlp_layers,
        mlp_hidden_dim,
        connectivity_radius,
        boundaries,
        normalization_stats,
        nparticle_types,
        particle_type_embedding_size,
        boundary_clamp_limit,
        device,
        rigid_bodies=[[0, 1]],
    )

    def fake_preprocessor(position_sequence, nparticles_per_example, particle_types, material_property=None):
        n = position_sequence.shape[0]
        node_features = torch.zeros(n, 14, dtype=position_sequence.dtype)
        edge_index = torch.zeros(2, 0, dtype=torch.long)
        edge_features = torch.zeros(0, 3, dtype=position_sequence.dtype)
        return node_features, edge_index, edge_features

    class FakeEPD(torch.nn.Module):
        def forward(self, node_features, edge_index, edge_features):
            return torch.zeros(node_features.shape[0], 2, dtype=node_features.dtype)

    captured = {}

    def fake_enforce(predicted_accelerations, positions, rigid_bodies, masses=None, eps=1e-6, velocities=None):
        captured["positions"] = positions.clone()
        captured["velocities"] = velocities.clone() if velocities is not None else None
        return predicted_accelerations

    monkeypatch.setattr(model, "_encoder_preprocessor", fake_preprocessor)
    monkeypatch.setattr(model, "_encode_process_decode", FakeEPD())
    monkeypatch.setattr("gns.learned_simulator.enforce_rigid_constraint", fake_enforce)

    position_sequence = torch.tensor(
        [
            [[0.0, 0.0], [1.0, 2.0]],
            [[2.0, 1.0], [5.0, 3.0]],
        ],
        dtype=torch.float32,
    )
    nparticles_per_example = torch.tensor([2], dtype=torch.long)
    particle_types = torch.zeros(2, dtype=torch.long)

    _ = model(position_sequence, nparticles_per_example, particle_types)

    expected_positions = position_sequence[:, -1, :]
    expected_velocities = position_sequence[:, -1, :] - position_sequence[:, -2, :]
    assert torch.allclose(captured["positions"], expected_positions)
    assert torch.allclose(captured["velocities"], expected_velocities)
