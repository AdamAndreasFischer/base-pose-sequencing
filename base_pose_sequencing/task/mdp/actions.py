from __future__ import annotations

import torch
import numpy as np
from scipy.spatial.transform import Rotation
from typing import TYPE_CHECKING
from dataclasses import MISSING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
import isaaclab.envs.mdp as mdp
import isaaclab.envs.mdp.actions.joint_actions as joint_actions
from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv




class MoveBaseAction(ActionTerm):
    """Base class for move base actions.
    """

    cfg: MoveBaseActionCfg
    """The configuration of the action term."""
    _asset: Articulation
    _scale: torch.Tensor | float
    """The scaling factor applied to the input action."""
    _clip: dict[str, float] 
    """The clip applied to the input action."""

    

    def __init__(self, cfg: MoveBaseActionCfg, env: ManagerBasedEnv) -> None:
        # initialize the action term
        super().__init__(cfg, env)

        # create tensors for raw and processed actions
        self._raw_actions = torch.zeros(self.num_envs, 3, device=self.device)
        self._processed_actions = torch.zeros(self.num_envs, 3, device=self.device)


    """
    Properties.
    """

    @property
    def action_dim(self) -> int:
        return 3

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    """
    Operations.
    """
    def process_actions(self, actions: torch.Tensor):
        """
        Process raw policy actions and clamp them to predefined limits.
        Policy outputs are assumed to be in the same physical units
        (meters for x,y, radians for theta).
        """
        self._raw_actions = actions.clone()
        print("In actions: ", actions)
        # Get limits from config
        lims = self.cfg.clip

        x_min, x_max = lims["x"]
        y_min, y_max = lims["y"]
        t_min, t_max = lims["theta"]

        x = torch.clamp(self._raw_actions[:, 0], x_min, x_max)
        y = torch.clamp(self._raw_actions[:, 1], y_min, y_max)
        theta = torch.clamp(self._raw_actions[:, 2], t_min, t_max)

        self._processed_actions = torch.stack([x, y, theta], dim=-1)

        


    def apply_actions(self):
        root_state = self._asset.data.root_state_w
       
        pos_x = self._processed_actions[:, 0]
        pos_y = self._processed_actions[:, 1]
        theta = self._processed_actions[:, 2]

        env_origins = self._env.scene.env_origins.to(self.device)   # (num_envs, 3)
        world_x = pos_x + env_origins[:, 0]
        world_y = pos_y + env_origins[:, 1]

        rot_quat = root_state[:, 3:7]
        rot_quat = np.roll(rot_quat.cpu(), -1, axis=1)

        rot_euler = Rotation.from_quat(rot_quat).as_euler('xyz', degrees=False)
        rot_euler[:, 2] = theta.cpu()

        rot = Rotation.from_euler('xyz', rot_euler, degrees=False)
        rot_quat = rot.as_quat()

        rot_quat = np.roll(rot_quat, 1, axis=1)
        rot_quat = torch.tensor(rot_quat).to(0)

        pos = torch.stack((world_x, world_y), dim=1)

        root_state[:, :2] = pos[:,:2].to(0)
        root_state[:, 3:7] = rot_quat

        self._asset.write_root_pose_to_sim(root_state[:, :7])



@configclass
class MoveBaseActionCfg(ActionTermCfg):
    """Configuration for the base joint action term.

    See :class:`MoveBaseAction` for more details.
    """
    class_type: type[ActionTerm] = MoveBaseAction

    scale: float | dict[str, float] = 1.0

    clip: dict[str, float]
