"""Additional reward for policy outputs that remain close to safe targets."""

from __future__ import annotations

import torch


def safety_target_correction_l2(env, action_name: str = "JointPositionAction") -> torch.Tensor:
    """Penalize the correction needed to enforce limits/rate constraints."""
    term = env.action_manager.get_term(action_name)
    return torch.sum(torch.square(term.requested_target - term.processed_actions), dim=-1)
