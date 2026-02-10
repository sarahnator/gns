"""Differentiable rigid body constraints for particle accelerations."""

from typing import List, Optional, Tuple

import torch
from torch import Tensor


def _cross_product_3d(a: Tensor, b: Tensor) -> Tensor:
    """Compute cross product a x b for 3D vectors."""
    return torch.cross(a, b, dim=-1)


def _cross_product_2d_scalar_vector(scalar: Tensor, v: Tensor) -> Tensor:
    """Compute omega x r in 2D where omega is the z-axis scalar."""
    return scalar * torch.stack([-v[:, 1], v[:, 0]], dim=-1)


def _compute_angular_velocity_3d(
    r: Tensor, velocities: Tensor, masses: Optional[Tensor], eps: float
) -> Tensor:
    """Estimate 3D angular velocity from relative positions and velocities."""
    if masses is None:
        mass_vec = torch.ones(r.shape[0], device=r.device, dtype=r.dtype)
    else:
        mass_vec = masses

    total_mass = mass_vec.sum()
    v_com = (mass_vec.unsqueeze(-1) * velocities).sum(dim=0) / total_mass
    v_rel = velocities - v_com

    inertia = compute_inertia_tensor_3d(r, mass_vec, eps)
    angular_momentum = _cross_product_3d(
        r, mass_vec.unsqueeze(-1) * v_rel
    ).sum(dim=0)
    return torch.linalg.solve(inertia, angular_momentum)


def _compute_angular_velocity_2d(
    r: Tensor, velocities: Tensor, masses: Optional[Tensor], eps: float
) -> Tensor:
    """Estimate 2D angular velocity (z scalar) from relative motion."""
    if masses is None:
        mass_vec = torch.ones(r.shape[0], device=r.device, dtype=r.dtype)
    else:
        mass_vec = masses

    total_mass = mass_vec.sum()
    v_com = (mass_vec.unsqueeze(-1) * velocities).sum(dim=0) / total_mass
    v_rel = velocities - v_com

    inertia = compute_inertia_scalar_2d(r, mass_vec, eps)
    angular_momentum_z = (
        r[:, 0] * (mass_vec * v_rel[:, 1]) - r[:, 1] * (mass_vec * v_rel[:, 0])
    ).sum()
    return angular_momentum_z / inertia


def compute_center_of_mass(
    positions: Tensor, masses: Optional[Tensor] = None
) -> Tuple[Tensor, Tensor]:
    """Compute center of mass and relative positions."""
    n = positions.shape[0]
    if masses is None:
        masses = torch.ones(n, device=positions.device, dtype=positions.dtype)

    total_mass = masses.sum()
    com = (masses.unsqueeze(-1) * positions).sum(dim=0) / total_mass
    r = positions - com
    return com, r


def compute_inertia_tensor_3d(r: Tensor, masses: Tensor, eps: float = 1e-6) -> Tensor:
    """Compute 3D inertia tensor about center of mass."""
    r_squared = (r * r).sum(dim=-1)
    outer = torch.einsum("ni,nj->nij", r, r)

    eye = torch.eye(3, device=r.device, dtype=r.dtype)
    identity_term = (masses * r_squared).sum() * eye
    outer_term = (masses.view(-1, 1, 1) * outer).sum(dim=0)
    inertia = identity_term - outer_term
    return inertia + eps * eye


def compute_inertia_scalar_2d(r: Tensor, masses: Tensor, eps: float = 1e-6) -> Tensor:
    """Compute 2D scalar inertia about center of mass."""
    r_squared = (r * r).sum(dim=-1)
    return (masses * r_squared).sum() + eps


def compute_rigid_body_state_3d(
    positions: Tensor,
    accelerations: Tensor,
    masses: Optional[Tensor] = None,
    eps: float = 1e-6,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Compute COM accel and angular accel for a 3D rigid body."""
    n = positions.shape[0]
    if masses is None:
        masses = torch.ones(n, device=positions.device, dtype=positions.dtype)

    total_mass = masses.sum()
    _, r = compute_center_of_mass(positions, masses)

    forces = masses.unsqueeze(-1) * accelerations
    net_force = forces.sum(dim=0)
    a_com = net_force / total_mass

    torque = _cross_product_3d(r, forces).sum(dim=0)
    inertia = compute_inertia_tensor_3d(r, masses, eps)
    angular_accel = torch.linalg.solve(inertia, torque)
    return a_com, angular_accel, r, inertia


def compute_rigid_body_state_2d(
    positions: Tensor,
    accelerations: Tensor,
    masses: Optional[Tensor] = None,
    eps: float = 1e-6,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Compute COM accel and angular accel for a 2D rigid body."""
    n = positions.shape[0]
    if masses is None:
        masses = torch.ones(n, device=positions.device, dtype=positions.dtype)

    total_mass = masses.sum()
    _, r = compute_center_of_mass(positions, masses)

    forces = masses.unsqueeze(-1) * accelerations
    net_force = forces.sum(dim=0)
    a_com = net_force / total_mass

    torque = (r[:, 0] * forces[:, 1] - r[:, 1] * forces[:, 0]).sum()
    inertia = compute_inertia_scalar_2d(r, masses, eps)
    angular_accel = torque / inertia
    return a_com, angular_accel, r, inertia


def reconstruct_particle_accelerations_3d(
    a_com: Tensor,
    angular_accel: Tensor,
    r: Tensor,
    angular_velocity: Optional[Tensor] = None,
) -> Tensor:
    """Reconstruct 3D per-particle accelerations from rigid-body state."""
    alpha = angular_accel.unsqueeze(0).expand(r.shape[0], -1)
    a_rot = _cross_product_3d(alpha, r)

    if angular_velocity is None:
        a_cent = torch.zeros_like(r)
    else:
        omega = angular_velocity.unsqueeze(0).expand(r.shape[0], -1)
        a_cent = _cross_product_3d(omega, _cross_product_3d(omega, r))

    return a_com + a_rot + a_cent


def reconstruct_particle_accelerations_2d(
    a_com: Tensor,
    angular_accel: Tensor,
    r: Tensor,
    angular_velocity: Optional[Tensor] = None,
) -> Tensor:
    """Reconstruct 2D per-particle accelerations from rigid-body state."""
    a_rot = _cross_product_2d_scalar_vector(angular_accel, r)
    if angular_velocity is None:
        a_cent = torch.zeros_like(r)
    else:
        a_cent = -(angular_velocity**2) * r
    return a_com + a_rot + a_cent


def enforce_rigid_constraint(
    predicted_accelerations: Tensor,
    positions: Tensor,
    rigid_bodies: List[Tensor],
    masses: Optional[Tensor] = None,
    eps: float = 1e-6,
    velocities: Optional[Tensor] = None,
) -> Tensor:
    """Project rigid-body particles to accelerations consistent with rigid motion.

    If `velocities` is provided, angular velocity is estimated from current
    rigid-body motion and centripetal acceleration is included. If omitted,
    behavior falls back to translation + angular-acceleration terms only.
    """
    if not rigid_bodies:
        return predicted_accelerations

    dim = positions.shape[-1]
    result = predicted_accelerations.clone()

    if dim == 3:
        compute_state = compute_rigid_body_state_3d
        reconstruct = reconstruct_particle_accelerations_3d
    elif dim == 2:
        compute_state = compute_rigid_body_state_2d
        reconstruct = reconstruct_particle_accelerations_2d
    else:
        raise ValueError(f"Unsupported dimension: {dim}. Must be 2 or 3.")

    for body_indices in rigid_bodies:
        if len(body_indices) < 2:
            continue
        body_indices = body_indices.to(positions.device)
        body_pos = positions[body_indices]
        body_accel = predicted_accelerations[body_indices]
        body_masses = masses[body_indices] if masses is not None else None
        body_velocities = velocities[body_indices] if velocities is not None else None

        a_com, angular_accel, r, _ = compute_state(body_pos, body_accel, body_masses, eps)
        if body_velocities is None:
            angular_velocity = None
        elif dim == 3:
            angular_velocity = _compute_angular_velocity_3d(
                r, body_velocities, body_masses, eps
            )
        else:
            angular_velocity = _compute_angular_velocity_2d(
                r, body_velocities, body_masses, eps
            )

        result[body_indices] = reconstruct(a_com, angular_accel, r, angular_velocity)


    return result


def compute_rigid_body_loss(
    predicted_accelerations: Tensor,
    ground_truth_accelerations: Tensor,
    positions: Tensor,
    rigid_bodies: List[Tensor],
    masses: Optional[Tensor] = None,
    eps: float = 1e-6,
) -> Tensor:
    """Optional DOF-level loss on rigid-body COM/angular accelerations."""
    if not rigid_bodies:
        return torch.tensor(
            0.0, device=predicted_accelerations.device, dtype=predicted_accelerations.dtype
        )

    dim = positions.shape[-1]
    compute_state = (
        compute_rigid_body_state_3d if dim == 3 else compute_rigid_body_state_2d
    )
    total_loss = torch.tensor(
        0.0, device=predicted_accelerations.device, dtype=predicted_accelerations.dtype
    )

    for body_indices in rigid_bodies:
        if len(body_indices) < 2:
            continue
        body_indices = body_indices.to(positions.device)
        body_pos = positions[body_indices]
        body_pred = predicted_accelerations[body_indices]
        body_gt = ground_truth_accelerations[body_indices]
        body_masses = masses[body_indices] if masses is not None else None

        a_com_pred, alpha_pred, _, _ = compute_state(body_pos, body_pred, body_masses, eps)
        a_com_gt, alpha_gt, _, _ = compute_state(body_pos, body_gt, body_masses, eps)

        loss_com = ((a_com_pred - a_com_gt) ** 2).sum()
        if dim == 3:
            loss_angular = ((alpha_pred - alpha_gt) ** 2).sum()
        else:
            loss_angular = (alpha_pred - alpha_gt) ** 2
        total_loss = total_loss + loss_com + loss_angular

    return total_loss
