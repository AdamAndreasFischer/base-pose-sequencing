from __future__ import annotations

import torch
from typing import TYPE_CHECKING
import warnings
import time
import copy
from isaaclab.assets import Articulation, RigidObject, RigidObjectCollection
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers.manager_base import ManagerTermBase
from isaaclab.managers.manager_term_cfg import RewardTermCfg
from isaaclab.sensors import ContactSensor, RayCaster
from isaaclab.managers import SceneEntityCfg
import isaaclab.utils.math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from base_pose_sequencing.utils.collision import check_if_robot_is_in_collision
from base_pose_sequencing.utils.common import parse_prim_paths
from base_pose_sequencing.utils.isaac import set_visibility_multi_object

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
    
    table_poses = table.data.body_link_state_w[env_ids].clone()
  
    obj_root_states = asset.data.default_object_state[env_ids].clone() # Initial states for the objects
    n_obj = obj_root_states.shape[1]

    # Table poses used as origin for randomized poses
    root_states = table_poses.repeat(1,n_obj,1) #Table root_state which is what we want as origin for objects

    range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset.device)#.unsqueeze(1).repeat(1,n_obj,1)
    
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), n_obj , 6), device=asset.device) #[N_env, N_obj, 6]
    
    # Z offset in order to spawn above table
    z_element = torch.tensor([[[0,0,0.32]]], device=env.device)
    z_offset = z_element.repeat(len(env_ids), n_obj,1)
   
    positions_local =  rand_samples[:,:, 0:3]+ z_offset # + origins

    # Transform poses with table rotation as to match x and y limits
    positions_rotated = math_utils.quat_apply(root_states[:,:,3:7], positions_local)
    
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
    
    
    object_prim_paths = env.scene["object"].root_physx_view.prim_paths
    ordered_paths = sorted(object_prim_paths, key=parse_prim_paths) # Prim paths ordered in after environment
    #asset.set_visibility(True, prim_paths=ordered_paths)
    #set_visibility_multi_object(True, prim_paths=ordered_paths)
    env.cfg.picked_objects = torch.zeros((obj_root_states.shape[0]*obj_root_states.shape[1]), device=env.device, dtype=torch.uint8)
    env.scene["camera"].reset(env_ids)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i in range(10):
            #env.sim.step(render=True)
            env.render(recompute=True)

   
# Currently not used. TODO: Add multi obstacle reset
def reset_obstacles(env: ManagerBasedRLEnv,
        env_ids: torch.Tensor,
        pose_range: dict[str, tuple[float, float]],
        velocity_range: dict[str, tuple[float, float]],
        asset_cfg: SceneEntityCfg = SceneEntityCfg("obstacle")):

    asset: RigidObject | Articulation | RigidObjectCollection = env.scene[asset_cfg.name]

    #root_states = asset.data.default_root_state[env_ids].clone()
    root_states = asset.data.default_object_state[env_ids].clone()

    n_obstacles = root_states.shape[1]
  
    # poses
    range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset.device)
    
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), n_obstacles, 6), device=asset.device)


    z_offset = torch.tensor([[0,0,0.18]]).repeat(len(env_ids), n_obstacles,1).to(env.device)

    origins = env.scene.env_origins[env_ids].repeat(1,n_obstacles, 1)

    positions =  origins + rand_samples[:, :, 0:3]+ z_offset
    orientations_delta = math_utils.quat_from_euler_xyz(rand_samples[:,:, 3], rand_samples[:,:, 4], rand_samples[:,:, 5])
    orientations = math_utils.quat_mul(root_states[:, 3:7], orientations_delta)
    # velocities
    vel_range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    vel_ranges = torch.tensor(vel_range_list, device=asset.device)
    rand_samples = math_utils.sample_uniform(vel_ranges[:, 0], vel_ranges[:, 1], (len(env_ids), n_obstacles, 6), device=asset.device)

    velocities = root_states[:, :, 7:13] + rand_samples

    # set into the physics simulation
    asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim(velocities, env_ids=env_ids)
    


def reset_robot_state(env: ManagerBasedRLEnv,
        env_ids: torch.Tensor,
        pose_range: dict[str, tuple[float, float]],
        velocity_range: dict[str, tuple[float, float]],
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    
    
    joint_indices = [3,4,7,8,9,10,11,12,13,14,16,16,17,18]
    joint_vel = torch.zeros_like(env.cfg.default_joint_angles, device=env.device)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    
    local_env_ids = copy.deepcopy(env_ids)
    not_suitable_pose = True
    while not_suitable_pose:
        origins = env.scene.env_origins[local_env_ids] # Robot randomization should use origin as center, not robot start pose
        root_states = asset.data.default_root_state[local_env_ids].clone()
        range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=asset.device)
        rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(local_env_ids), 6), device=asset.device)

        positions = origins+rand_samples[:,:3] 
        orientations_delta = math_utils.quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
        orientations = math_utils.quat_mul(root_states[:, 3:7], orientations_delta)
        # velocities
        vel_range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        vel_ranges = torch.tensor(vel_range_list, device=asset.device)
        rand_samples = math_utils.sample_uniform(vel_ranges[:, 0], vel_ranges[:, 1], (len(local_env_ids), 6), device=asset.device)

        velocities = root_states[ :, 7:13] + rand_samples
        #print("Pre set pose")
        # set into the physics simulation

        asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=local_env_ids)
        asset.write_root_velocity_to_sim(velocities, env_ids=local_env_ids)
        # Test not stepping environment before reset of sensor as to not push robot away from collision
        #for _ in range(2):
        #    env.sim.step(render=True)

        env.scene["contact_forces_robot"].reset(local_env_ids)
        for _ in range(2): #This stepping is crucial as it updates the physics of the simulation
            env.sim.step(render=True)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            forces_world = env.scene["contact_forces_robot"].data.force_matrix_w 
        forces_env = forces_world[local_env_ids].squeeze(1) # Squeeze to remove dimenison that show ammount of bodies in sensor (?. I think it corresponds to how many meshes are connected to it?
        
        force_detection = (forces_env==0.).all(dim=(1,2)).to(dtype=torch.bool)

        if torch.any(force_detection!=True):
        
            local_env_ids = local_env_ids[(~force_detection)]
            
            
        else:
            not_suitable_pose = False
    #asset.write_joint_state_to_sim(position= env.cfg.default_joint_angles, velocity = joint_vel, joint_ids= joint_indices, env_ids = None)
        

def reset_obstacles_singular(env: ManagerBasedRLEnv,
        env_ids: torch.Tensor,
        pose_range: dict[str, tuple[float, float]],
        velocity_range: dict[str, tuple[float, float]],
        asset_cfg: SceneEntityCfg = SceneEntityCfg("obstacle")):
   
    asset: RigidObject | Articulation | RigidObjectCollection = env.scene[asset_cfg.name]


    #root_states = asset.data.default_root_state[env_ids].clone()
    root_states = asset.data.default_root_state[env_ids].clone()

    # poses
    range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset.device)
    
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=asset.device)

    z_offset = torch.tensor([[0,0,0.18]]).repeat(len(env_ids),1).to(env.device)

    origins = env.scene.env_origins[env_ids]

    positions =  origins + rand_samples[ :, 0:3]+ z_offset
    orientations_delta = math_utils.quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
    orientations = math_utils.quat_mul(root_states[:, 3:7], orientations_delta)
    # velocities
    vel_range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    vel_ranges = torch.tensor(vel_range_list, device=asset.device)
    rand_samples = math_utils.sample_uniform(vel_ranges[:, 0], vel_ranges[:, 1], (len(env_ids), 6), device=asset.device)

    velocities = root_states[ :, 7:13] + rand_samples

    # set into the physics simulation
    asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim(velocities, env_ids=env_ids)





def reset_table(
    env: ManagerBasedRLEnv,
        env_ids: torch.Tensor,
        pose_range: dict[str, tuple[float, float]],
        velocity_range: dict[str, tuple[float, float]],
        asset_cfg: SceneEntityCfg = SceneEntityCfg("table")):
    
    asset: RigidObject | Articulation | RigidObjectCollection = env.scene[asset_cfg.name]

    # poses
    range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset.device)
    local_env_ids = copy.deepcopy(env_ids)
    not_suitable_pose = True

    while not_suitable_pose:
        root_states = asset.data.default_root_state[local_env_ids].clone()
        rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(local_env_ids), 6), device=asset.device)


        z_offset = torch.tensor([[0,0,0.28]]).repeat(len(local_env_ids),1).to(env.device)

        origins = env.scene.env_origins[local_env_ids]
    
        positions =  origins + rand_samples[ :, 0:3]+ z_offset
        orientations_delta = math_utils.quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
        orientations = math_utils.quat_mul(root_states[:, 3:7], orientations_delta)

        # velocities
        vel_range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        vel_ranges = torch.tensor(vel_range_list, device=asset.device)

        rand_samples = math_utils.sample_uniform(vel_ranges[:, 0], vel_ranges[:, 1], (len(local_env_ids), 6), device=asset.device)

        velocities = root_states[ :, 7:13] + rand_samples
   
        # set into the physics simulation
        asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=local_env_ids)
        asset.write_root_velocity_to_sim(velocities, env_ids=local_env_ids)

            
    
        env.scene["contact_forces_table"].reset([local_env_ids]) #Resetting the sensor seems to update it as intended! Debug results, two iterations with contact shows force before and after reset, 
        
        for _ in range(2):
            env.sim.step(render=True)
        
        forces_world = env.scene["contact_forces_table"].data.force_matrix_w 
        forces_env = forces_world[local_env_ids].squeeze(1) # Squeeze to remove dimenison that show ammount of bodies in sensor (?. I think it corresponds to how many meshes are connected to it?
    
        force_detection = (forces_env==0.).all(dim=(1,2)).to(dtype=torch.bool)
    

        if torch.any(force_detection!=True):
        
            local_env_ids = local_env_ids[(~force_detection)]
            
        else:
            not_suitable_pose = False
               


def zero_velocities(env: ManagerBasedRLEnv,
                    env_ids: torch.Tensor):
    """
    Function that zeros velocity of table and robot, and orients it correctly incase it has a velocity after reset
    """
    table = env.scene["table"]
    robot = env.scene["robot"]

    table_pose = table.data.root_link_state_w[env_ids]
    robot_pose = robot.data.root_link_state_w[env_ids]

    euler_table = math_utils.euler_xyz_from_quat(table_pose[:,3:7])
    euler_robot = math_utils.euler_xyz_from_quat(robot_pose[:,3:7])
    
    table_vel_shape = table_pose[:,7:13]

    yaw_table = euler_table[2]
    yaw_robot = euler_robot[2]

    zeros = torch.zeros_like(yaw_table)

    table_quat = math_utils.quat_from_euler_xyz(zeros,zeros,yaw_table)
    robot_quat = math_utils.quat_from_euler_xyz(zeros,zeros,yaw_robot)
    
    robot.write_root_pose_to_sim(torch.cat([robot_pose[:,:3], robot_quat],dim=-1), env_ids=env_ids)
    table.write_root_pose_to_sim(torch.cat([table_pose[:,:3], table_quat],dim=-1), env_ids=env_ids)

    table.write_root_velocity_to_sim(torch.zeros_like(table_vel_shape),env_ids=env_ids)
    robot.write_root_velocity_to_sim(torch.zeros_like(table_vel_shape),env_ids=env_ids)

    env.cfg.prev_robot_poses = robot.data.root_link_state_w

