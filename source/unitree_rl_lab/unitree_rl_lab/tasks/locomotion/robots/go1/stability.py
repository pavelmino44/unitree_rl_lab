"""Body-pose safeguards for Go1 flat locomotion (training only).

The contact sensor sees collisions of URDF collision bodies, not decorative
visual mesh. Thus a low/nose-down robot may not trigger trunk contact.
Keep fall detection independent of contact forces.
"""

from __future__ import annotations

import torch


def _base_z(env) -> torch.Tensor:
    # This is the world height of the articulation root link. On flat terrain
    # with ground z=0, it is a stable proxy for the trunk body height.
    return env.scene["robot"].data.root_link_pos_w[:, 2]


def base_too_low(env, minimum_height: float) -> torch.Tensor:
    """End an episode when root link height drops below a strict floor."""
    return _base_z(env) < minimum_height


def low_base_height_penalty(env, target_height: float) -> torch.Tensor:
    """Penalize body positions *below* nominal height; don't reward jumping."""
    return torch.clamp(target_height - _base_z(env), min=0.0)
