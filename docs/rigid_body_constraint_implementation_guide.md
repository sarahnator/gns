# Rigid Body Constraint Implementation Guide

## Overview

This document provides a complete implementation guide for integrating differentiable rigid body constraints into Graph Network-based Simulators (GNS). The approach constrains predicted accelerations to produce physically valid rigid body motion without using Kabsch alignment or shape matching.

**Key Principle:** The network predicts per-particle accelerations. For rigid body particles, we project these onto the space of valid rigid motions (translation + rotation). Gradients flow through this projection, so the network learns to predict forces that produce correct rigid body dynamics.

---

## Table of Contents

1. [Mathematical Foundation](#1-mathematical-foundation)
2. [Implementation Architecture](#2-implementation-architecture)
3. [Core Algorithm](#3-core-algorithm)
4. [PyTorch Implementation](#4-pytorch-implementation)
5. [Integration with GNS](#5-integration-with-gns)
6. [Loss Computation](#6-loss-computation)
7. [Handling 2D vs 3D](#7-handling-2d-vs-3d)
8. [Configuration Options](#8-configuration-options)
9. [Testing Strategy](#9-testing-strategy)
10. [Common Pitfalls](#10-common-pitfalls)

---

## 1. Mathematical Foundation

### 1.1 Rigid Body State

A rigid body's motion is characterized by 6 degrees of freedom (in 3D):
- **3 translational DOF:** Position/velocity/acceleration of center of mass
- **3 rotational DOF:** Orientation/angular velocity/angular acceleration

All particles in a rigid body move coherently according to:

```
x_i(t) = x_com(t) + R(t) · r_i^0
```

Where:
- `x_i(t)` = position of particle i at time t
- `x_com(t)` = center of mass position
- `R(t)` = rotation matrix (orientation)
- `r_i^0` = particle's position relative to COM in rest configuration

### 1.2 Acceleration Constraint

For accelerations, the relationship is:

```
a_i = a_com + α × r_i + ω × (ω × r_i)
```

Where:
- `a_i` = acceleration of particle i
- `a_com` = acceleration of center of mass
- `α` = angular acceleration
- `ω` = angular velocity
- `r_i` = current position of particle i relative to COM

**Simplification:** When constraining predicted accelerations (not integrating dynamics), we use:

```
a_i = a_com + α × r_i
```

The centripetal term `ω × (ω × r_i)` is omitted because:
1. We don't track angular velocity explicitly
2. The network's predictions already account for current motion state
3. For small timesteps, this term is second-order

### 1.3 Computing Rigid Body State from Particle Accelerations

Given predicted accelerations for particles in a rigid body, we extract the rigid body state:

**Step 1: Center of Mass Acceleration**
```
a_com = (Σ m_i · a_i) / M

where M = Σ m_i (total mass)
```

This is equivalent to: net force on body = M · a_com

**Step 2: Net Torque about COM**
```
τ = Σ r_i × F_i = Σ r_i × (m_i · a_i)

where r_i = x_i - x_com
```

**Step 3: Moment of Inertia Tensor**
```
I = Σ m_i · (|r_i|² · I₃ - r_i ⊗ r_i)

where:
- I₃ = 3×3 identity matrix
- r_i ⊗ r_i = outer product (r_i · r_i^T)
```

**Step 4: Angular Acceleration**
```
α = I⁻¹ · τ
```

With regularization for numerical stability:
```
α = (I + ε·I₃)⁻¹ · τ

where ε ≈ 1e-6
```

### 1.4 Reconstructing Per-Particle Accelerations

Once we have `a_com` and `α`, compute constrained accelerations:

```
a_i^constrained = a_com + α × r_i
```

This ensures all particles move as a rigid unit.

---

## 2. Implementation Architecture

### 2.1 Data Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           TRAINING FORWARD PASS                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Input: positions, velocities, particle_types, edges                     │
│                          │                                               │
│                          ▼                                               │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                    GNN Message Passing                            │   │
│  │  - Node encoder                                                   │   │
│  │  - Edge encoder                                                   │   │
│  │  - Message passing layers                                         │   │
│  │  - Node decoder                                                   │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                          │                                               │
│                          ▼                                               │
│           predicted_accelerations: (N, dim)                              │
│           [May be inconsistent for rigid particles]                      │
│                          │                                               │
│                          ▼                                               │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │              RIGID BODY CONSTRAINT (Differentiable)               │   │
│  │                                                                   │   │
│  │  For each rigid body:                                             │   │
│  │    1. Extract particle positions and predicted accelerations      │   │
│  │    2. Compute COM acceleration: a_com = Σ(m_i·a_i) / M           │   │
│  │    3. Compute torque: τ = Σ r_i × (m_i·a_i)                      │   │
│  │    4. Compute inertia tensor: I = Σ m_i(|r|²I - r⊗r)            │   │
│  │    5. Compute angular acceleration: α = I⁻¹τ                     │   │
│  │    6. Reconstruct: a_i = a_com + α × r_i                         │   │
│  │                                                                   │   │
│  │  Non-rigid particles: unchanged                                   │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                          │                                               │
│                          ▼                                               │
│           constrained_accelerations: (N, dim)                            │
│           [Physically consistent rigid motion]                           │
│                          │                                               │
│                          ▼                                               │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                      LOSS COMPUTATION                             │   │
│  │                                                                   │   │
│  │  Option A: Per-particle MSE                                       │   │
│  │    loss = MSE(constrained_accel, ground_truth)                    │   │
│  │                                                                   │   │
│  │  Option B: Rigid body state loss                                  │   │
│  │    loss = MSE(a_com_pred, a_com_gt) + MSE(α_pred, α_gt)          │   │
│  │                                                                   │   │
│  │  Kinematic particles excluded from loss                           │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                          │                                               │
│                          ▼                                               │
│                     Backpropagation                                      │
│           [Gradients flow through constraint]                            │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| When to apply constraint | Forward pass (training + inference) | Consistency between training and rollout |
| What to constrain | Accelerations | Network's actual output; forces are accelerations × mass |
| Angular velocity estimation | Not used | Constraining accelerations, not integrating dynamics |
| Inertia regularization | ε = 1e-6 | Prevents singular matrix for degenerate configurations |
| Non-rigid particles | Pass through unchanged | Only modify particles belonging to rigid bodies |

---

## 3. Core Algorithm

### 3.1 Pseudocode

```
function enforce_rigid_constraint(predicted_accel, positions, rigid_bodies, masses):
    result = copy(predicted_accel)

    for body_indices in rigid_bodies:
        # Extract data for this body
        pos = positions[body_indices]          # (n_body, dim)
        accel = predicted_accel[body_indices]  # (n_body, dim)
        m = masses[body_indices]               # (n_body,)

        # Compute center of mass
        M = sum(m)
        com = sum(m * pos) / M
        r = pos - com  # relative positions

        # Compute COM acceleration (from net force)
        F = m * accel  # forces on each particle
        F_net = sum(F)
        a_com = F_net / M

        # Compute net torque about COM
        if dim == 3:
            τ = sum(cross(r, F))  # (3,)
        else:  # dim == 2
            τ = sum(r[:,0]*F[:,1] - r[:,1]*F[:,0])  # scalar

        # Compute moment of inertia
        if dim == 3:
            I = sum(m * (|r|² * I₃ - outer(r, r)))  # (3, 3)
        else:
            I = sum(m * |r|²)  # scalar

        # Compute angular acceleration
        α = solve(I + ε*I, τ)  # I⁻¹τ with regularization

        # Reconstruct per-particle accelerations
        if dim == 3:
            a_rot = cross(α, r)  # (n_body, 3)
        else:
            a_rot = α * [-r[:,1], r[:,0]]  # (n_body, 2)

        constrained_accel = a_com + a_rot

        # Write back
        result[body_indices] = constrained_accel

    return result
```

### 3.2 Complexity Analysis

For a rigid body with `n` particles:
- COM computation: O(n)
- Torque computation: O(n)
- Inertia tensor: O(n)
- Matrix solve: O(1) (3×3 system)
- Reconstruction: O(n)

**Total: O(n) per rigid body**

For `k` rigid bodies with total `N` rigid particles:
**Total: O(N)**

The constraint adds negligible overhead compared to GNN message passing.

---

## 4. PyTorch Implementation

### 4.1 Complete Module: `gns/rigid_body.py`

```python
"""
Differentiable rigid body constraint for GNS.

This module provides functions to constrain predicted particle accelerations
to valid rigid body motion. The constraint is fully differentiable, allowing
gradients to flow through during training.

Key functions:
- enforce_rigid_constraint: Main entry point for applying constraint
- compute_rigid_body_state: Extract COM accel and angular accel from particles
- reconstruct_particle_accelerations: Convert rigid state back to per-particle
- compute_rigid_body_loss: Optional loss directly on rigid body DOFs
"""

import torch
from torch import Tensor
from typing import List, Optional, Tuple, Union


def _skew_symmetric_3d(v: Tensor) -> Tensor:
    """
    Compute skew-symmetric matrix for cross product.

    For vector v = [x, y, z], returns matrix S such that S @ u = v × u

    Args:
        v: Vector of shape (..., 3)

    Returns:
        Skew-symmetric matrix of shape (..., 3, 3)
    """
    # Handle batched input
    batch_shape = v.shape[:-1]
    x, y, z = v.unbind(-1)

    zero = torch.zeros_like(x)

    # Build skew-symmetric matrix
    row1 = torch.stack([zero, -z, y], dim=-1)
    row2 = torch.stack([z, zero, -x], dim=-1)
    row3 = torch.stack([-y, x, zero], dim=-1)

    return torch.stack([row1, row2, row3], dim=-2)


def _cross_product_3d(a: Tensor, b: Tensor) -> Tensor:
    """
    Compute cross product a × b for 3D vectors.

    Args:
        a: Shape (*, 3) or (3,)
        b: Shape (*, 3) or (3,)

    Returns:
        Cross product, shape (*, 3)
    """
    return torch.cross(a, b, dim=-1)


def _cross_product_2d_scalar_vector(scalar: Tensor, v: Tensor) -> Tensor:
    """
    Compute ω × r in 2D where ω is a scalar (z-component of angular velocity).

    In 2D, if ω = [0, 0, ω_z] and r = [r_x, r_y, 0], then:
    ω × r = [−ω_z·r_y, ω_z·r_x, 0]

    Args:
        scalar: Angular velocity/acceleration, shape () or (1,)
        v: Position vectors, shape (n, 2)

    Returns:
        Result of cross product, shape (n, 2)
    """
    # ω × r = [-ω*r_y, ω*r_x]
    return scalar * torch.stack([-v[:, 1], v[:, 0]], dim=-1)


def compute_center_of_mass(
    positions: Tensor,
    masses: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor]:
    """
    Compute center of mass and relative positions.

    Args:
        positions: Particle positions, shape (n, dim)
        masses: Optional masses, shape (n,). Defaults to uniform.

    Returns:
        com: Center of mass, shape (dim,)
        r: Relative positions, shape (n, dim)
    """
    n = positions.shape[0]
    device = positions.device

    if masses is None:
        masses = torch.ones(n, device=device)

    total_mass = masses.sum()

    # COM = Σ(m_i * x_i) / M
    com = (masses.unsqueeze(-1) * positions).sum(dim=0) / total_mass

    # Relative positions
    r = positions - com

    return com, r


def compute_inertia_tensor_3d(
    r: Tensor,
    masses: Tensor,
    eps: float = 1e-6,
) -> Tensor:
    """
    Compute 3D moment of inertia tensor about center of mass.

    I = Σ m_i * (|r_i|² * I₃ - r_i ⊗ r_i)

    Args:
        r: Relative positions from COM, shape (n, 3)
        masses: Particle masses, shape (n,)
        eps: Regularization for numerical stability

    Returns:
        Inertia tensor with regularization, shape (3, 3)
    """
    device = r.device
    n = r.shape[0]

    # |r_i|² for each particle
    r_squared = (r * r).sum(dim=-1)  # (n,)

    # Outer products r_i ⊗ r_i: (n, 3, 3)
    outer = torch.einsum('ni,nj->nij', r, r)

    # Identity contribution: m_i * |r_i|² * I₃
    eye = torch.eye(3, device=device)
    identity_term = (masses * r_squared).sum() * eye

    # Outer product contribution: -Σ m_i * (r_i ⊗ r_i)
    outer_term = (masses.view(n, 1, 1) * outer).sum(dim=0)

    # Full inertia tensor
    inertia = identity_term - outer_term

    # Add regularization
    inertia = inertia + eps * eye

    return inertia


def compute_inertia_scalar_2d(
    r: Tensor,
    masses: Tensor,
    eps: float = 1e-6,
) -> Tensor:
    """
    Compute 2D moment of inertia (scalar) about center of mass.

    I = Σ m_i * |r_i|²

    Args:
        r: Relative positions from COM, shape (n, 2)
        masses: Particle masses, shape (n,)
        eps: Regularization for numerical stability

    Returns:
        Scalar moment of inertia (with regularization)
    """
    r_squared = (r * r).sum(dim=-1)  # |r_i|² for each particle
    inertia = (masses * r_squared).sum() + eps
    return inertia


def compute_rigid_body_state_3d(
    positions: Tensor,
    accelerations: Tensor,
    masses: Optional[Tensor] = None,
    eps: float = 1e-6,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    Compute 3D rigid body state from particle data.

    Args:
        positions: Particle positions, shape (n, 3)
        accelerations: Predicted accelerations, shape (n, 3)
        masses: Optional masses, shape (n,)
        eps: Regularization for inertia tensor

    Returns:
        a_com: COM acceleration, shape (3,)
        angular_accel: Angular acceleration, shape (3,)
        r: Relative positions from COM, shape (n, 3)
        inertia: Inertia tensor, shape (3, 3)
    """
    n = positions.shape[0]
    device = positions.device

    if masses is None:
        masses = torch.ones(n, device=device)

    total_mass = masses.sum()

    # Center of mass and relative positions
    com, r = compute_center_of_mass(positions, masses)

    # Forces on each particle: F_i = m_i * a_i
    forces = masses.unsqueeze(-1) * accelerations  # (n, 3)

    # Net force and COM acceleration
    net_force = forces.sum(dim=0)  # (3,)
    a_com = net_force / total_mass  # (3,)

    # Net torque about COM: τ = Σ r_i × F_i
    torque = _cross_product_3d(r, forces).sum(dim=0)  # (3,)

    # Moment of inertia tensor
    inertia = compute_inertia_tensor_3d(r, masses, eps)  # (3, 3)

    # Angular acceleration: α = I⁻¹ τ
    angular_accel = torch.linalg.solve(inertia, torque)  # (3,)

    return a_com, angular_accel, r, inertia


def compute_rigid_body_state_2d(
    positions: Tensor,
    accelerations: Tensor,
    masses: Optional[Tensor] = None,
    eps: float = 1e-6,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    Compute 2D rigid body state from particle data.

    Args:
        positions: Particle positions, shape (n, 2)
        accelerations: Predicted accelerations, shape (n, 2)
        masses: Optional masses, shape (n,)
        eps: Regularization for inertia

    Returns:
        a_com: COM acceleration, shape (2,)
        angular_accel: Angular acceleration, scalar tensor
        r: Relative positions from COM, shape (n, 2)
        inertia: Scalar moment of inertia
    """
    n = positions.shape[0]
    device = positions.device

    if masses is None:
        masses = torch.ones(n, device=device)

    total_mass = masses.sum()

    # Center of mass and relative positions
    com, r = compute_center_of_mass(positions, masses)

    # Forces on each particle
    forces = masses.unsqueeze(-1) * accelerations  # (n, 2)

    # Net force and COM acceleration
    net_force = forces.sum(dim=0)  # (2,)
    a_com = net_force / total_mass  # (2,)

    # Net torque about COM (scalar, z-component)
    # τ_z = Σ (r_x * F_y - r_y * F_x)
    torque = (r[:, 0] * forces[:, 1] - r[:, 1] * forces[:, 0]).sum()

    # Moment of inertia (scalar)
    inertia = compute_inertia_scalar_2d(r, masses, eps)

    # Angular acceleration: α = τ / I
    angular_accel = torque / inertia

    return a_com, angular_accel, r, inertia


def reconstruct_particle_accelerations_3d(
    a_com: Tensor,
    angular_accel: Tensor,
    r: Tensor,
) -> Tensor:
    """
    Reconstruct per-particle accelerations from rigid body state (3D).

    a_i = a_com + α × r_i

    Args:
        a_com: COM acceleration, shape (3,)
        angular_accel: Angular acceleration, shape (3,)
        r: Relative positions, shape (n, 3)

    Returns:
        Per-particle accelerations, shape (n, 3)
    """
    n = r.shape[0]

    # Expand angular acceleration for cross product with each r_i
    alpha_expanded = angular_accel.unsqueeze(0).expand(n, -1)  # (n, 3)

    # Rotational acceleration: α × r_i
    a_rot = _cross_product_3d(alpha_expanded, r)  # (n, 3)

    # Total acceleration
    return a_com + a_rot


def reconstruct_particle_accelerations_2d(
    a_com: Tensor,
    angular_accel: Tensor,
    r: Tensor,
) -> Tensor:
    """
    Reconstruct per-particle accelerations from rigid body state (2D).

    a_i = a_com + α × r_i

    In 2D: α × r = [-α*r_y, α*r_x]

    Args:
        a_com: COM acceleration, shape (2,)
        angular_accel: Angular acceleration, scalar
        r: Relative positions, shape (n, 2)

    Returns:
        Per-particle accelerations, shape (n, 2)
    """
    # Rotational acceleration
    a_rot = _cross_product_2d_scalar_vector(angular_accel, r)  # (n, 2)

    # Total acceleration
    return a_com + a_rot


def enforce_rigid_constraint(
    predicted_accelerations: Tensor,
    positions: Tensor,
    rigid_bodies: List[Tensor],
    masses: Optional[Tensor] = None,
    eps: float = 1e-6,
) -> Tensor:
    """
    Apply rigid body constraint to predicted accelerations.

    This is the main entry point. It projects per-particle accelerations
    onto the space of valid rigid body motions. The operation is fully
    differentiable.

    Args:
        predicted_accelerations: Network output, shape (n_total, dim)
        positions: Current positions, shape (n_total, dim)
        rigid_bodies: List of 1D tensors, each containing indices of
                      particles belonging to one rigid body
        masses: Optional masses, shape (n_total,)
        eps: Regularization for inertia computation

    Returns:
        Constrained accelerations, shape (n_total, dim)
        - Rigid body particles: consistent with rigid motion
        - Non-rigid particles: unchanged

    Example:
        >>> positions = torch.randn(100, 3)
        >>> accelerations = torch.randn(100, 3, requires_grad=True)
        >>> rigid_bodies = [torch.tensor([0,1,2,3]), torch.tensor([50,51,52])]
        >>> constrained = enforce_rigid_constraint(accelerations, positions, rigid_bodies)
        >>> loss = constrained.sum()
        >>> loss.backward()  # Gradients flow through
    """
    if not rigid_bodies:
        return predicted_accelerations

    dim = positions.shape[-1]
    device = positions.device

    # Clone to avoid modifying input
    result = predicted_accelerations.clone()

    # Select appropriate functions based on dimensionality
    if dim == 3:
        compute_state = compute_rigid_body_state_3d
        reconstruct = reconstruct_particle_accelerations_3d
    elif dim == 2:
        compute_state = compute_rigid_body_state_2d
        reconstruct = reconstruct_particle_accelerations_2d
    else:
        raise ValueError(f"Unsupported dimension: {dim}. Must be 2 or 3.")

    for body_indices in rigid_bodies:
        # Skip degenerate bodies
        if len(body_indices) < 2:
            continue

        # Ensure indices are on correct device
        body_indices = body_indices.to(device)

        # Extract data for this body
        body_pos = positions[body_indices]
        body_accel = predicted_accelerations[body_indices]
        body_masses = masses[body_indices] if masses is not None else None

        # Compute rigid body state
        a_com, angular_accel, r, _ = compute_state(
            body_pos, body_accel, body_masses, eps
        )

        # Reconstruct constrained accelerations
        constrained = reconstruct(a_com, angular_accel, r)

        # Write back (in-place modification of clone)
        result[body_indices] = constrained

    return result


def compute_rigid_body_loss(
    predicted_accelerations: Tensor,
    ground_truth_accelerations: Tensor,
    positions: Tensor,
    rigid_bodies: List[Tensor],
    masses: Optional[Tensor] = None,
    eps: float = 1e-6,
) -> Tensor:
    """
    Compute loss directly on rigid body degrees of freedom.

    Instead of penalizing each particle redundantly, this computes loss
    on the 6 DOFs per body (3D) or 3 DOFs (2D):
    - COM acceleration (3 or 2 components)
    - Angular acceleration (3 or 1 component)

    Args:
        predicted_accelerations: Constrained predictions, shape (n, dim)
        ground_truth_accelerations: Ground truth, shape (n, dim)
        positions: Current positions, shape (n, dim)
        rigid_bodies: List of body index tensors
        masses: Optional masses
        eps: Regularization

    Returns:
        Scalar loss tensor (sum of squared errors on rigid body state)
    """
    if not rigid_bodies:
        return torch.tensor(0.0, device=predicted_accelerations.device)

    dim = positions.shape[-1]
    device = positions.device

    if dim == 3:
        compute_state = compute_rigid_body_state_3d
    else:
        compute_state = compute_rigid_body_state_2d

    total_loss = torch.tensor(0.0, device=device)

    for body_indices in rigid_bodies:
        if len(body_indices) < 2:
            continue

        body_indices = body_indices.to(device)
        body_pos = positions[body_indices]
        body_pred = predicted_accelerations[body_indices]
        body_gt = ground_truth_accelerations[body_indices]
        body_masses = masses[body_indices] if masses is not None else None

        # Extract state from predictions
        a_com_pred, alpha_pred, _, _ = compute_state(
            body_pos, body_pred, body_masses, eps
        )

        # Extract state from ground truth
        a_com_gt, alpha_gt, _, _ = compute_state(
            body_pos, body_gt, body_masses, eps
        )

        # Loss on COM acceleration
        loss_com = ((a_com_pred - a_com_gt) ** 2).sum()

        # Loss on angular acceleration
        if dim == 3:
            loss_angular = ((alpha_pred - alpha_gt) ** 2).sum()
        else:
            loss_angular = (alpha_pred - alpha_gt) ** 2

        total_loss = total_loss + loss_com + loss_angular

    return total_loss


def get_rigid_body_state(
    predicted_accelerations: Tensor,
    positions: Tensor,
    rigid_bodies: List[Tensor],
    masses: Optional[Tensor] = None,
    eps: float = 1e-6,
) -> List[dict]:
    """
    Extract rigid body state for visualization/debugging.

    Args:
        predicted_accelerations: Predictions, shape (n, dim)
        positions: Positions, shape (n, dim)
        rigid_bodies: List of body index tensors
        masses: Optional masses
        eps: Regularization

    Returns:
        List of dicts, one per body, containing:
        - 'com': center of mass position
        - 'a_com': COM acceleration
        - 'angular_accel': angular acceleration
        - 'inertia': moment of inertia
        - 'indices': particle indices
    """
    dim = positions.shape[-1]
    device = positions.device

    if dim == 3:
        compute_state = compute_rigid_body_state_3d
    else:
        compute_state = compute_rigid_body_state_2d

    results = []

    for body_indices in rigid_bodies:
        if len(body_indices) < 2:
            continue

        body_indices = body_indices.to(device)
        body_pos = positions[body_indices]
        body_accel = predicted_accelerations[body_indices]
        body_masses = masses[body_indices] if masses is not None else None

        com, _ = compute_center_of_mass(body_pos, body_masses)
        a_com, angular_accel, r, inertia = compute_state(
            body_pos, body_accel, body_masses, eps
        )

        results.append({
            'com': com.detach(),
            'a_com': a_com.detach(),
            'angular_accel': angular_accel.detach(),
            'inertia': inertia.detach(),
            'indices': body_indices.detach(),
        })

    return results
```

---

## 5. Integration with GNS

### 5.1 Modify `learned_simulator.py`

```python
# At top of file
from gns.rigid_body import enforce_rigid_constraint, compute_rigid_body_loss

class LearnedSimulator(nn.Module):
    def __init__(
        self,
        particle_dimensions: int,
        # ... existing args ...
        rigid_bodies: Optional[List[List[int]]] = None,
    ):
        super().__init__()
        # ... existing initialization ...

        # Store rigid body configuration
        self._rigid_bodies = None
        if rigid_bodies is not None:
            self._rigid_bodies = [
                torch.tensor(indices, dtype=torch.long)
                for indices in rigid_bodies
            ]

    @property
    def rigid_bodies(self) -> Optional[List[Tensor]]:
        """Get rigid body index tensors (moved to appropriate device)."""
        return self._rigid_bodies

    def set_rigid_bodies(self, rigid_bodies: List[List[int]]):
        """Set rigid bodies dynamically (e.g., from data loader)."""
        self._rigid_bodies = [
            torch.tensor(indices, dtype=torch.long)
            for indices in rigid_bodies
        ]

    def forward(
        self,
        position_sequence: Tensor,
        n_particles_per_example: Tensor,
        particle_types: Tensor,
        material_property: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Forward pass with rigid body constraint.

        Returns constrained accelerations.
        """
        # ... existing preprocessing ...
        # ... existing GNN forward ...

        predicted_accelerations = self._decoder(node_features)

        # Apply rigid body constraint (differentiable)
        if self._rigid_bodies is not None:
            # Move indices to same device as data
            device = predicted_accelerations.device
            rigid_bodies_on_device = [
                idx.to(device) for idx in self._rigid_bodies
            ]

            # Get current positions (last timestep)
            current_positions = position_sequence[:, -1, :]

            predicted_accelerations = enforce_rigid_constraint(
                predicted_accelerations,
                current_positions,
                rigid_bodies_on_device,
                masses=None,  # Uniform masses; or pass actual masses
            )

        return predicted_accelerations
```

### 5.2 Modify Training Loop

```python
# In train.py or equivalent

def train_step(
    simulator: LearnedSimulator,
    batch: Tuple,
    optimizer: torch.optim.Optimizer,
    kinematic_mask: Optional[Tensor] = None,
    use_rigid_body_loss: bool = False,
):
    """
    Single training step with rigid body support.
    """
    positions, particle_types, targets, ... = batch

    optimizer.zero_grad()

    # Forward pass (constraint applied inside)
    predicted = simulator(positions, particle_types, ...)

    # Compute loss
    if use_rigid_body_loss and simulator.rigid_bodies:
        # Option B: Loss on rigid body state
        current_positions = positions[:, -1, :]

        # Per-particle loss for non-rigid particles
        non_rigid_mask = torch.ones(positions.shape[0], dtype=torch.bool)
        if kinematic_mask is not None:
            non_rigid_mask = non_rigid_mask & ~kinematic_mask
        for body_indices in simulator.rigid_bodies:
            non_rigid_mask[body_indices] = False

        loss_particles = F.mse_loss(
            predicted[non_rigid_mask],
            targets[non_rigid_mask]
        )

        # Rigid body state loss
        loss_rigid = compute_rigid_body_loss(
            predicted, targets, current_positions,
            simulator.rigid_bodies
        )

        loss = loss_particles + loss_rigid
    else:
        # Option A: Standard per-particle loss on constrained output
        if kinematic_mask is not None:
            loss_mask = ~kinematic_mask
            loss = F.mse_loss(predicted[loss_mask], targets[loss_mask])
        else:
            loss = F.mse_loss(predicted, targets)

    loss.backward()
    optimizer.step()

    return loss.item()
```

### 5.3 Rollout (No Changes Needed)

Since the constraint is part of `forward()`, rollout works automatically:

```python
# In render_rollout.py or equivalent

def rollout(simulator, initial_positions, num_steps, dt, ...):
    """
    Rollout simulation. Rigid body constraint applied automatically.
    """
    positions = initial_positions.clone()
    velocities = torch.zeros_like(positions)

    trajectory = [positions.clone()]

    for step in range(num_steps):
        # Forward pass includes rigid constraint
        predicted_accel = simulator(
            position_sequence,
            n_particles_per_example,
            particle_types,
        )

        # Semi-implicit Euler integration
        velocities = velocities + predicted_accel * dt
        positions = positions + velocities * dt

        trajectory.append(positions.clone())

    return torch.stack(trajectory)
```

---

## 6. Loss Computation

### 6.1 Option A: Per-Particle MSE (Simple)

```python
# Constraint already applied in forward()
loss = F.mse_loss(predicted[~kinematic_mask], ground_truth[~kinematic_mask])
```

**Pros:**
- Simple to implement
- Works with existing training infrastructure

**Cons:**
- Redundant: penalizes N particles for 6 DOFs of error
- May over-weight rigid bodies proportional to particle count

### 6.2 Option B: Rigid Body State Loss (Recommended)

```python
# Per-particle loss for non-rigid particles
loss_particles = F.mse_loss(predicted[non_rigid_mask], ground_truth[non_rigid_mask])

# Rigid body loss (6 DOF per body)
loss_rigid = compute_rigid_body_loss(predicted, ground_truth, positions, rigid_bodies)

# Combine (may need weighting)
loss = loss_particles + lambda_rigid * loss_rigid
```

**Pros:**
- Directly supervises on the actual DOFs
- No redundancy
- Balanced contribution regardless of particle count

**Cons:**
- Slightly more complex
- May need to tune `lambda_rigid`

### 6.3 Weighting Considerations

If using Option B, consider the scale of each loss component:

```python
# Number of non-rigid particles
n_non_rigid = non_rigid_mask.sum()

# Number of rigid DOFs (6 per body in 3D)
n_rigid_dof = 6 * len(rigid_bodies)

# Weight to balance contributions
lambda_rigid = n_non_rigid / n_rigid_dof
```

---

## 7. Handling 2D vs 3D

### 7.1 Differences

| Aspect | 2D | 3D |
|--------|----|----|
| DOFs per body | 3 (x, y, θ) | 6 (x, y, z, θx, θy, θz) |
| Angular acceleration | Scalar | 3D vector |
| Inertia | Scalar | 3×3 tensor |
| Torque | Scalar (z-component) | 3D vector |
| Cross product | `α * [-r_y, r_x]` | Standard `α × r` |

### 7.2 Automatic Handling

The implementation automatically detects dimensionality:

```python
dim = positions.shape[-1]

if dim == 3:
    # Use 3D functions
    compute_state = compute_rigid_body_state_3d
    reconstruct = reconstruct_particle_accelerations_3d
elif dim == 2:
    # Use 2D functions
    compute_state = compute_rigid_body_state_2d
    reconstruct = reconstruct_particle_accelerations_2d
```

### 7.3 2D Cross Product Details

In 2D, angular velocity/acceleration is a scalar (rotation about z-axis):

```
ω = [0, 0, ω_z]  →  represented as scalar ω_z
α = [0, 0, α_z]  →  represented as scalar α_z
```

The cross product `α × r` where `r = [r_x, r_y, 0]`:

```
α × r = [0, 0, α_z] × [r_x, r_y, 0]
      = [-α_z * r_y, α_z * r_x, 0]
      → [-α * r_y, α * r_x]  (2D result)
```

---

## 8. Configuration Options

### 8.1 Static Configuration (config.yaml)

```yaml
# Rigid bodies defined by particle indices
rigid_bodies:
  - name: "block_1"
    indices: [0, 1, 2, 3, 4, 5, 6, 7]
  - name: "block_2"
    indices: [100, 101, 102, 103, 104, 105, 106, 107]

# Or by particle type (all particles of type X form one body)
rigid_body_types: [3]  # All type-3 particles are rigid

# Loss configuration
rigid_body_loss:
  use_state_loss: true  # Use 6-DOF loss vs per-particle
  weight: 1.0           # Weight for rigid body loss component
```

### 8.2 Dynamic from Data

If rigid bodies vary per sample, load from data files:

```python
# In data loader
def load_sample(path):
    data = np.load(path)
    positions = data['positions']
    body_ids = data['body_id']  # -1 for non-rigid

    # Convert body_ids to list of index tensors
    rigid_bodies = []
    for bid in np.unique(body_ids):
        if bid >= 0:  # Skip sentinel
            indices = np.where(body_ids == bid)[0]
            rigid_bodies.append(torch.tensor(indices))

    return positions, rigid_bodies
```

### 8.3 Runtime Configuration

```python
# Create simulator without rigid bodies
simulator = LearnedSimulator(...)

# Set rigid bodies dynamically
rigid_bodies = [[0, 1, 2, 3], [10, 11, 12, 13]]
simulator.set_rigid_bodies(rigid_bodies)
```

---

## 9. Testing Strategy

### 9.1 Unit Tests

**Test 1: Shape Preservation**
```python
def test_shape_preservation():
    """Constrained motion should preserve inter-particle distances."""
    positions = torch.tensor([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
    accelerations = torch.randn(3, 3)
    rigid_bodies = [torch.tensor([0, 1, 2])]

    constrained = enforce_rigid_constraint(accelerations, positions, rigid_bodies)

    # Apply accelerations for small dt
    dt = 0.001
    new_pos = positions + 0.5 * constrained * dt**2

    # Check distances preserved
    d_before = (positions[0] - positions[1]).norm()
    d_after = (new_pos[0] - new_pos[1]).norm()

    assert torch.allclose(d_before, d_after, atol=1e-4)
```

**Test 2: Differentiability**
```python
def test_gradients_flow():
    """Gradients should flow through the constraint."""
    positions = torch.randn(10, 3)
    accelerations = torch.randn(10, 3, requires_grad=True)
    rigid_bodies = [torch.tensor([0, 1, 2, 3, 4])]

    constrained = enforce_rigid_constraint(accelerations, positions, rigid_bodies)
    loss = constrained.sum()
    loss.backward()

    assert accelerations.grad is not None
    assert accelerations.grad.abs().sum() > 0
```

**Test 3: Pure Translation**
```python
def test_pure_translation():
    """Uniform acceleration should give pure translation."""
    positions = torch.tensor([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
    uniform_accel = torch.tensor([[1., 2., 3.]]).expand(3, -1).clone()
    rigid_bodies = [torch.tensor([0, 1, 2])]

    constrained = enforce_rigid_constraint(uniform_accel, positions, rigid_bodies)

    # Should be unchanged (already valid rigid motion)
    assert torch.allclose(constrained, uniform_accel, atol=1e-6)
```

**Test 4: Non-Rigid Unchanged**
```python
def test_non_rigid_unchanged():
    """Non-rigid particles should not be modified."""
    positions = torch.randn(10, 3)
    accelerations = torch.randn(10, 3)
    rigid_bodies = [torch.tensor([0, 1, 2])]  # Only first 3 are rigid

    constrained = enforce_rigid_constraint(accelerations, positions, rigid_bodies)

    # Particles 3-9 should be unchanged
    assert torch.allclose(constrained[3:], accelerations[3:])
```

**Test 5: 2D Support**
```python
def test_2d_constraint():
    """Should work for 2D simulations."""
    positions = torch.randn(5, 2)
    accelerations = torch.randn(5, 2, requires_grad=True)
    rigid_bodies = [torch.tensor([0, 1, 2])]

    constrained = enforce_rigid_constraint(accelerations, positions, rigid_bodies)

    assert constrained.shape == (5, 2)

    loss = constrained.sum()
    loss.backward()
    assert accelerations.grad is not None
```

### 9.2 Integration Tests

```python
def test_training_with_rigid_bodies():
    """Full training loop with rigid bodies."""
    simulator = LearnedSimulator(
        particle_dimensions=3,
        rigid_bodies=[[0, 1, 2, 3, 4]],
        # ... other args ...
    )

    optimizer = torch.optim.Adam(simulator.parameters())

    for _ in range(10):
        positions = torch.randn(20, 6, 3)  # 20 particles, 6 timesteps
        targets = torch.randn(20, 3)

        predicted = simulator(positions, ...)
        loss = F.mse_loss(predicted, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Should complete without errors
    assert True
```

### 9.3 Physics Validation

```python
def test_angular_momentum_conservation():
    """For isolated body with no net torque, angular momentum conserved."""
    # Set up body with zero net torque
    positions = torch.tensor([[-1., 0., 0.], [1., 0., 0.]])

    # Equal and opposite forces (zero net force, zero net torque)
    accelerations = torch.tensor([[1., 0., 0.], [-1., 0., 0.]])

    rigid_bodies = [torch.tensor([0, 1])]
    constrained = enforce_rigid_constraint(accelerations, positions, rigid_bodies)

    # COM acceleration should be zero
    a_com = constrained.mean(dim=0)
    assert torch.allclose(a_com, torch.zeros(3), atol=1e-6)
```

---

## 10. Common Pitfalls

### 10.1 Singular Inertia Tensor

**Problem:** Collinear particles produce singular inertia tensor.

**Solution:** Regularization (already included):
```python
inertia = inertia + eps * torch.eye(3)
```

### 10.2 Device Mismatch

**Problem:** Rigid body indices on CPU, data on GPU.

**Solution:** Move indices in forward pass:
```python
rigid_bodies_on_device = [idx.to(device) for idx in self._rigid_bodies]
```

### 10.3 In-Place Modification

**Problem:** Modifying input tensor breaks gradient computation.

**Solution:** Clone before modification:
```python
result = predicted_accelerations.clone()
result[body_indices] = constrained
```

### 10.4 Single-Particle Bodies

**Problem:** Body with one particle has undefined rotation.

**Solution:** Skip degenerate bodies:
```python
if len(body_indices) < 2:
    continue
```

### 10.5 Forgetting to Apply Constraint

**Problem:** Applying constraint only at inference, not training.

**Solution:** Put constraint in `forward()`, not separate post-processing.

### 10.6 Wrong Loss Mask

**Problem:** Computing loss on raw predictions instead of constrained.

**Solution:** Loss computed after constraint in forward pass.

---

## Appendix A: Quick Start Checklist

- [ ] Create `gns/rigid_body.py` with constraint functions
- [ ] Modify `LearnedSimulator.__init__` to accept `rigid_bodies`
- [ ] Modify `LearnedSimulator.forward` to apply constraint
- [ ] Verify loss uses constrained predictions (automatic if constraint in forward)
- [ ] Add unit tests
- [ ] Test with both 2D and 3D simulations
- [ ] Verify gradients flow (check `param.grad` is not None/zero)

## Appendix B: Performance Considerations

- Constraint adds O(N) operations where N = number of rigid particles
- Matrix solve is O(1) per body (3×3 system)
- Negligible compared to GNN message passing
- No need for optimization unless > 100 rigid bodies

## Appendix C: Extension Ideas

1. **Per-particle masses:** Pass actual masses instead of uniform
2. **Articulated bodies:** Chain of rigid bodies with joints
3. **Soft constraints:** Blend constrained and unconstrained (for quasi-rigid)
4. **Dynamic body detection:** Cluster particles into rigid bodies at runtime
