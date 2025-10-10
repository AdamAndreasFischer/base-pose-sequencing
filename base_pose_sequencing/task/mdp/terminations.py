from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def collision_check(env:ManagerBasedRLEnv, robot: SceneEntityCfg = SceneEntityCfg("robot")):

    colision= torch.tensor([0]).to(env.device)
    return colision