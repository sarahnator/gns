import os
import sys

import torch

# Add parent directory to Python path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

from gns.rigid_body import enforce_rigid_constraint


def test_constraint_2d_changes_only_rigid_particles():
    positions = torch.tensor(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [5.0, 5.0]], dtype=torch.float32
    )
    accelerations = torch.tensor(
        [[0.3, -0.2], [0.8, 0.1], [-0.5, 0.9], [1.2, -0.7]], dtype=torch.float32
    )
    rigid_bodies = [torch.tensor([0, 1, 2], dtype=torch.long)]

    constrained = enforce_rigid_constraint(accelerations, positions, rigid_bodies)

    assert constrained.shape == accelerations.shape
    assert not torch.allclose(constrained[:3], accelerations[:3])
    assert torch.allclose(constrained[3], accelerations[3])


def test_constraint_2d_adds_centripetal_term_from_velocities():
    positions = torch.tensor(
        [[1.0, 0.0], [-1.0, 0.0], [4.0, 4.0]], dtype=torch.float32
    )
    # Rigid body particles rotating around origin with omega=2 rad/s.
    velocities = torch.tensor(
        [[0.0, 2.0], [0.0, -2.0], [0.0, 0.0]], dtype=torch.float32
    )
    accelerations = torch.zeros_like(positions)
    rigid_bodies = [torch.tensor([0, 1], dtype=torch.long)]

    constrained = enforce_rigid_constraint(
        accelerations, positions, rigid_bodies, velocities=velocities
    )

    # a = -omega^2 * r = -4 * r
    expected = torch.tensor([[-4.0, 0.0], [4.0, 0.0]], dtype=torch.float32)
    assert torch.allclose(constrained[:2], expected, atol=1e-4)
    assert torch.allclose(constrained[2], accelerations[2])


def test_constraint_3d_runs():
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [2.0, 2.0, 2.0]],
        dtype=torch.float32,
    )
    accelerations = torch.tensor(
        [[0.2, -0.1, 0.4], [0.3, 0.9, -0.2], [-0.7, 0.5, 0.6], [1.0, 1.0, 1.0]],
        dtype=torch.float32,
    )
    rigid_bodies = [torch.tensor([0, 1, 2], dtype=torch.long)]

    constrained = enforce_rigid_constraint(accelerations, positions, rigid_bodies)

    assert constrained.shape == accelerations.shape
    assert torch.allclose(constrained[3], accelerations[3])


def test_constraint_3d_adds_centripetal_term_from_velocities():
    positions = torch.tensor(
        [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [3.0, 3.0, 3.0]], dtype=torch.float32
    )
    # Rotation about z-axis with omega=[0,0,2].
    velocities = torch.tensor(
        [[0.0, 2.0, 0.0], [0.0, -2.0, 0.0], [0.0, 0.0, 0.0]], dtype=torch.float32
    )
    accelerations = torch.zeros_like(positions)
    rigid_bodies = [torch.tensor([0, 1], dtype=torch.long)]

    constrained = enforce_rigid_constraint(
        accelerations, positions, rigid_bodies, velocities=velocities
    )

    expected = torch.tensor(
        [[-4.0, 0.0, 0.0], [4.0, 0.0, 0.0]], dtype=torch.float32
    )
    assert torch.allclose(constrained[:2], expected, atol=1e-4)
    assert torch.allclose(constrained[2], accelerations[2])


def test_constraint_without_velocities_matches_previous_behavior():
    positions = torch.tensor(
        [[1.0, 0.0], [-1.0, 0.0], [2.0, 2.0]], dtype=torch.float32
    )
    accelerations = torch.zeros_like(positions)
    rigid_bodies = [torch.tensor([0, 1], dtype=torch.long)]

    constrained = enforce_rigid_constraint(accelerations, positions, rigid_bodies)
    assert torch.allclose(constrained[:2], torch.zeros(2, 2))


def test_gradients_flow_2d():
    positions = torch.randn(6, 2)
    accelerations = torch.randn(6, 2, requires_grad=True)
    rigid_bodies = [torch.tensor([0, 1, 2, 3], dtype=torch.long)]

    constrained = enforce_rigid_constraint(accelerations, positions, rigid_bodies)
    loss = constrained.sum()
    loss.backward()

    assert accelerations.grad is not None
    assert torch.any(accelerations.grad != 0)


def test_gradients_flow_3d():
    positions = torch.randn(7, 3)
    accelerations = torch.randn(7, 3, requires_grad=True)
    rigid_bodies = [torch.tensor([1, 2, 3, 4], dtype=torch.long)]

    constrained = enforce_rigid_constraint(accelerations, positions, rigid_bodies)
    loss = constrained.pow(2).sum()
    loss.backward()

    assert accelerations.grad is not None
    assert torch.any(accelerations.grad != 0)
