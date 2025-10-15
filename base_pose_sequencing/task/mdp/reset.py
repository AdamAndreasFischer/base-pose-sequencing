from __future__ import annotations

import torch
from typing import TYPE_CHECKING
import warnings
import time

from isaaclab.assets import Articulation, RigidObject, RigidObjectCollection
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers.manager_base import ManagerTermBase
from isaaclab.managers.manager_term_cfg import RewardTermCfg
from isaaclab.sensors import ContactSensor, RayCaster
from isaaclab.managers import SceneEntityCfg
import isaaclab.utils.math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def reset_object_state_uniform(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    table_cfg: SceneEntityCfg= SceneEntityCfg("table")
):
    """Reset the asset root state to a random position and velocity uniformly within the given ranges.

    This function randomizes the root position and velocity of the asset.

    * It samples the root position from the given ranges and adds them to the default root position, before setting
      them into the physics simulation.
    * It samples the root orientation from the given ranges and sets them into the physics simulation.
    * It samples the root velocity from the given ranges and sets them into the physics simulation.

    The function takes a dictionary of pose and velocity ranges for each axis and rotation. The keys of the
    dictionary are ``x``, ``y``, ``z``, ``roll``, ``pitch``, and ``yaw``. The values are tuples of the form
    ``(min, max)``. If the dictionary does not contain a key, the position or velocity is set to zero for that axis.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation | RigidObjectCollection = env.scene[asset_cfg.name]
   
   
    # get default root state
    table: RigidObject = env.scene[table_cfg.name]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        table_poses = table.data.body_state_w
  
    obj_root_states = asset.data.default_object_state[env_ids].clone() # Initial states for the objects
    n_obj = obj_root_states.shape[1]

    root_states = table_poses.repeat(1,n_obj,1) #Table root_state which is what we want as origin for objects

    
    # poses
    range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset.device)#.unsqueeze(1).repeat(1,n_obj,1)
    
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), n_obj , 6), device=asset.device) #[N_env, N_obj, 6]
    
    origins = env.scene.env_origins[env_ids].unsqueeze(1).repeat(1,n_obj,1)# Scene origins
    
    z_element = torch.tensor([[[0,0,0.32]]], device=env.device)
    z_offset = z_element.repeat(root_states.shape[0], root_states.shape[1],1)

    positions_local =  rand_samples[:,:, 0:3]+ z_offset # + origins

    positions_rotated = math_utils.quat_rotate(root_states[:,:,3:7], positions_local)
    
    positions = root_states[:,:, 0:3] +positions_rotated

    orientations_delta = math_utils.quat_from_euler_xyz(rand_samples[:,:, 3], rand_samples[:,:, 4], rand_samples[:,:, 5])
    orientations = math_utils.quat_mul(root_states[:,:, 3:7], orientations_delta)
    
    # velocities
    range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), n_obj,6), device=asset.device)

    velocities = root_states[:, :, 7:13] + rand_samples

    # set into the physics simulation
    asset.write_object_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
    asset.write_object_velocity_to_sim(velocities, env_ids=env_ids)

def reset_obstacles(env: ManagerBasedRLEnv,
        env_ids: torch.Tensor,
        pose_range: dict[str, tuple[float, float]],
        velocity_range: dict[str, tuple[float, float]],
        asset_cfg: SceneEntityCfg = SceneEntityCfg("obstacle")):

    asset: RigidObject | Articulation | RigidObjectCollection = env.scene[asset_cfg.name]

    root_states = asset.data.default_root_state[env_ids].clone()

    # poses
    range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset.device)
    
    not_suitable_pose = True
    #while not_suitable_pose:

    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=asset.device)
    #root_states[:, 0:3] + 
    
    z_offset = torch.tensor([[0,0,0.18]]).repeat(len(env_ids),1).to(env.device)

    positions = env.scene.env_origins[env_ids] + rand_samples[:, 0:3]+ z_offset
    orientations_delta = math_utils.quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
    orientations = math_utils.quat_mul(root_states[:, 3:7], orientations_delta)
    # velocities
    vel_range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    vel_ranges = torch.tensor(vel_range_list, device=asset.device)
    rand_samples = math_utils.sample_uniform(vel_ranges[:, 0], vel_ranges[:, 1], (len(env_ids), 6), device=asset.device)

    velocities = root_states[:, 7:13] + rand_samples

    # set into the physics simulation
    asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim(velocities, env_ids=env_ids)
    
    #env.scene.update(env.scene.physics_dt, sensor_update= True) # TEST update scene in order to get correct redings
 
    # Using force sensor for reset collision detection does not work as the env must be stepped to update physics calculations
        #with warnings.catch_warnings():
        #    warnings.simplefilter("ignore")
        #    forces_world = env.scene["contact_forces_obstacle"].data.force_matrix_w 
        #forces_env = forces_world[env_ids]
#
        #xy_force = torch.abs(forces_env[:, :, :2])
        #
        #mask = (xy_force>0.1).any(dim=1)
        #if torch.any(mask):
        #    print("collision")
        #    print(xy_force)
        #    print(positions)
        #    time.sleep(1)
        #else:
        #    not_suitable_pose = False