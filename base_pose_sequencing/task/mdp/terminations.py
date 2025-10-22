from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def collision_check(env:ManagerBasedRLEnv,threshold: float):
    """
    Check if robot is in collision with either obstacle or table
    """
    
    robot_forces = env.scene["contact_forces_robot"].data.force_matrix_w
    robot_forces= robot_forces.squeeze(1) #Removing wierd dim that shows how many bodies has the sensor (?)

    xy_forces = robot_forces[:,:,:2].abs()
    contact_mask = (xy_forces>threshold).any(dim=-1)
    collision = contact_mask.any(dim=-1)

    return collision