
import pytorch_kinematics as pk
import torch
import numpy as np
from scipy.ndimage import binary_dilation
from scipy.spatial.transform import Rotation
from typing import TYPE_CHECKING
import matplotlib.pyplot as plt
import collections
import time
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers.manager_base import ManagerTermBase
from isaaclab.managers.manager_term_cfg import RewardTermCfg
from isaaclab.sensors import ContactSensor, RayCaster
from isaaclab.managers import SceneEntityCfg
import isaacsim.core.utils.bounds as bounds_utils
from isaaclab.utils.math import euler_xyz_from_quat

from base_pose_sequencing.utils.collision import check_if_robot_is_in_collision
from base_pose_sequencing.utils.torch_kinematics import _summarize_results
from base_pose_sequencing.utils.torch_kinematics import assemble_full_configuration, compute_dual_arm_end_effector_poses, _summarize_results, get_robot_IK, get_robot_chains
from base_pose_sequencing.utils.common import parse_prim_paths, pose_to_pixel, visualize_scene, compare_pixel_to_world, world_to_pixel
from base_pose_sequencing.task.mdp.terminations import collision_check
from base_pose_sequencing.utils.isaac import set_visibility_multi_object
from base_pose_sequencing.utils.navigation import parallel_Astar, a_star, visualize_path, a_star_numpy

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def collision(env: "ManagerBasedRLEnv",
              robot_cfg: SceneEntityCfg= SceneEntityCfg("robot"),
              ):
    """Checks for collisions and updates termination manager (env.termination_manager.terminated)"""

    collisions = collision_check(env, threshold=100).to(torch.uint8)
    #print(env.termination_manager.terminated)
    #env.termination_manager.terminated = collisions.to(torch.bool)


    # return 1 for collision, 0 if no collision

    return collisions

def get_costmap(env, camera, env_ids, clearance):
    costmaps = np.zeros((len(env_ids), 160,160))
    segmentation_map = env.scene["camera"].data.output["semantic_segmentation"][env_ids].squeeze(-1).cpu().numpy()
    camera_info = camera.data.info
    structure = np.ones((2*clearance+1, 2*clearance+1),dtype=bool)
    
    for i in range(segmentation_map.shape[0]):
        idToLabels= camera_info[i]["semantic_segmentation"]["idToLabels"]
        #print(f"env {i}: ", idToLabels)
        costmap = costmaps[i]
        for key,value in idToLabels.items():
         
            if "table" in value or "obstacle" in value:
        
                costmap[segmentation_map[i]==int(key)] = 1
        costmap = binary_dilation(costmap>0, structure=structure)
        costmaps[i] = costmap
    
    return costmaps

def navcost(env: "ManagerBasedRLEnv",
            ):
    
    """
    Function for calculating the navigation cost of the action. What is needed is to access the previous robot pose, the current robot pose, and with a path planner 
    decide what the cost of the base pose selection is. 

    args: 
    goals: torch.tensor [n_envs, 2] x,y coordinates of the goal
    

    TODO: Fix variable for initial robot pose
    Fix path planner 
    Costmap extraction with obstacles

    Parallel A* can find paths for 40 envs in roughly 3.5 s, as compared to 36 for sequential
    

    Filters out collided environments
    """

    #compare_pixel_to_world(env=env)
    camera = env.scene["camera"]
    robot = env.scene["robot"]
    env_ids_raw = torch.arange(0, env.num_envs).to(device=env.device)
    env_ids = env_ids_raw[env.termination_manager.terminated==False] # removes the calculation for collided environments
    
    #Goal poses
    raw_robot_poses = robot.data.root_link_state_w[env_ids,:2] #x, y pose
    robot_rotations = robot.data.root_link_state_w[env_ids, 3:7]
    euler = euler_xyz_from_quat(robot_rotations)
    z_rot_goal = euler[-1]
    
    origin_poses = env.scene.env_origins[env_ids,:2]

    robot_goals = raw_robot_poses-origin_poses # Current pose is the goal in the calculation
    goal_x = robot_goals[:,0]
    goal_y= robot_goals[:,1]

    observation_space = env.observation_space["policy"]

    start_pose = env.cfg.prev_robot_pose
    start_trans = start_pose[env_ids,:2] - origin_poses # Previous robot pose is the start
    
    start_x = start_trans[:,0]
    start_y = start_trans[:,1]
    start_quats = start_pose[env_ids, 3:7]
    start_euler = euler_xyz_from_quat(start_quats)
    z_rot_start = start_euler[-1]

    #starts = pose_to_pixel(env, observation_space=observation_space, poses=start_pose)
    starts = world_to_pixel(env,start_x, start_y, z_rot_start,env_ids=env_ids)

    #goals = pose_to_pixel(env,observation_space, poses=robot_goals)
    goals = world_to_pixel(env, goal_x, goal_y, z_rot_goal, env_ids=env_ids)
  
    costmaps = get_costmap(env, camera, env_ids,clearance=8)
    
  
    path_list = []
 
    for i in range(costmaps.shape[0]):
        start = (starts[i,0].item(), starts[i,1].item())
        goal = (goals[i,0].item(), goals[i,1].item())
        
        costmap = costmaps[i]#.cpu().numpy()
        path,_ = a_star(costmap, start, goal, clearance=8, id=i)
        path_list.append(path)

    #visualize_path(costmaps[0], path_list[0], (starts[0,0].cpu(), starts[0,1].cpu()), (goals[0,0].cpu(), goals[0,1].cpu()), env , 0)

    rewards = torch.ones((env.num_envs), device=env.device)*250 # Base reward for no nav path found. I.e collided envs and non path found
 
    for i,env_id in enumerate(env_ids):
        if path_list[i] == None:
            rewards[env_id] =250
          
        else:
            rewards[env_id] = len(path_list[i])
    
    return rewards


def pick_reward(env: "ManagerBasedRLEnv",
                ):
    """
    Reward for picking objects. This needs a kinematic solver which can process multiple poses at the same time, possibly torch IK solver. 
    TODO: Function for calculating grasp poses
    Function of choosing arm for picking

    """
    terminated = env.termination_manager.terminated
    objects = env.scene["object"]
    obj_poses = objects.data.object_link_state_w[..., :7] # C.O.M states
    reward = torch.zeros((env.num_envs),device=env.device)
    # Initialization of picked_objects
    if env.cfg.picked_objects.shape[0]!=obj_poses.shape[0]*obj_poses.shape[1]:
        env.cfg.picked_objects = torch.zeros((obj_poses.shape[0]*obj_poses.shape[1]))

    # Get crucial functions for the robots
    robot_conf={}
    robot = env.scene["robot"]
    left_chain, right_chain, robot_chain = get_robot_chains(env.device)
    left_IK, right_IK, left_default, right_default = get_robot_IK(env.device) # Base frame for IK solvers is base_link, i.e the ridgebacks position must be used
    
    right_indices = [4,  8, 10,12, 14, 16, 18]
    left_indices = [3,  7, 9,  11, 13, 15, 17]

    robot_conf["right_chain"]   = right_chain
    robot_conf["left_chain"]    = left_chain
    robot_conf["robot_chain"]   = robot_chain
    robot_conf["left_default"]  = left_default
    robot_conf["right_default"] = right_default

    
    # Picked objects. O if not picked, 1 if picked
    picked_objects = env.cfg.picked_objects
    #object_prim_paths = env.scene["object"].root_physx_view.prim_paths

    obj_indices = torch.arange(0,picked_objects.shape[0], device=env.device, dtype=torch.uint8) # Gives an index to each prim path in the list
    #ordered_paths = sorted(object_prim_paths, key=parse_prim_paths) # Prim paths ordered in after environment

    # Initialize poses
    
    robot_pose = robot.data.root_state_w[..., :7] # Root state, i.e ridgeback root state [20,7] xyz qw qx qy qz
   
    n_obj = obj_poses.shape[1]
    
    env_ids = torch.arange(0,obj_poses.shape[0]).unsqueeze(-1).to(device=env.device)

    env_ids_expanded_og = env_ids.repeat_interleave(n_obj, dim=0)

    origins = env.scene.env_origins[env_ids]

    obj_poses_flattened = obj_poses.reshape(-1, obj_poses.shape[-1]) 

    # Select arms for grasping 
    arm_id = select_robot_arm_for_grasping(poses_global=obj_poses_flattened, robot_poses_global=robot_pose, origin_poses=origins, robot_conf=robot_conf, device=env.device, n_obj=n_obj)
    
    # Get graps poses
    _, grasp_matrices = get_grasp_pose(poses_global=obj_poses_flattened, n_obj=n_obj, origins=origins, robot_poses=robot_pose, device=env.device)

    # Remove already picked objects from consideration
    arm_id = arm_id[picked_objects==0]
    
    grasp_matrices = grasp_matrices[picked_objects==0]
    
    obj_indices = obj_indices[picked_objects==0]
    
    env_ids_expanded = env_ids_expanded_og[picked_objects==0]
 

    # Split up targets poses for their corresponding manipulator

    left_mask = (arm_id==1)
    right_mask = (arm_id==0)

    l_target_poses = grasp_matrices[left_mask]
    r_target_poses = grasp_matrices[right_mask]

    l_envs = env_ids_expanded[left_mask]
    r_envs = env_ids_expanded[right_mask]

    r_indices = obj_indices[right_mask]
    l_indices = obj_indices[left_mask]

    exists_l = False
    exists_r= False

    # Find IK solutions for both arms

    if l_target_poses.shape[0]>0:
        l_targets = pk.transform3d.Transform3d(default_batch_size=l_target_poses.shape[0], matrix=l_target_poses, device=env.device)

        solution = left_IK.solve(
            target_poses=l_targets
        )
        _,left_configs = _summarize_results(left_chain, solution, targets= l_targets, arm_name="left")
        action_l = left_configs
        success_l = solution.converged_any
        left_success_env_id = l_envs[success_l]
        l_picked_indices = l_indices[success_l]

        #print("Successes left: ", success_l)
        #print("Success env ids left: ", left_success_env_id)
        if torch.any(success_l!=False).item():
             exists_l=True
        
    if r_target_poses.shape[0]>0:
        r_targets = pk.transform3d.Transform3d(default_batch_size=r_target_poses.shape[0], matrix=r_target_poses, device=env.device)
        solution = right_IK.solve(target_poses=r_targets)
        _,right_configs = _summarize_results(right_chain, solution, targets=r_targets, arm_name="right")
        action_r = right_configs
        success_r = solution.converged_any
        right_success_env_id = r_envs[success_r]
        r_picked_indices = r_indices[success_r]

        #print("Successes right: ", success_r)
        #print("Success env ids right: ", right_success_env_id)
        if torch.any(success_r!=False).item():
             exists_r=True
    
    # For visualization. Move arms to found joint angles
    if exists_r: # Batch joint angles
        velocities = torch.zeros_like(action_r[0])
        for i,action in enumerate(action_r):
            env_id = right_success_env_id[i]
  
            env.scene["robot"].write_joint_state_to_sim(position= action, velocity = velocities, joint_ids= right_indices, env_ids = env_id.unsqueeze(0))
            #for i in range(1):
            #    env.sim.render()
    far_away_pose = torch.tensor([[[99.0, 99.0, 1.0, 0.0, 0.0, 0.0, 1.0]]], device=env.device) # (1,1,7)
    if exists_r:    
        for index in r_picked_indices:
            #print(index)
            env_id = env_ids_expanded_og[index.item()]
            env.cfg.picked_objects[index.item()] = 1
        
            object_index = index % objects.num_objects  # Get object index within environment
            object_index = object_index.to(dtype=torch.int64)
        
            objects.write_object_pose_to_sim(far_away_pose, env_ids=env_id, object_ids=object_index.reshape(1)) # cant be done in batch as it move all objects of same indice in each environment
            reward[env_id] +=1

    
    if exists_l:
        velocities = torch.zeros_like(action_l[0])
        for i,action in enumerate(action_l):
            env_id = left_success_env_id[i]
       
            env.scene["robot"].write_joint_state_to_sim(position= action, velocity = velocities ,joint_ids= left_indices, env_ids=env_id.unsqueeze(0))
            #for i in range(1):
            #    env.sim.render()
    if exists_l:    
        for index in l_picked_indices:
            #print(index)
            env_id = env_ids_expanded_og[index.item()]
            env.cfg.picked_objects[index.item()] = 1

            # Move picked object far away instead of hiding
            object_index = (index % objects.num_objects)  # Get object index within environment
            object_index = object_index.to(dtype=torch.int64)
          
            objects.write_object_pose_to_sim(far_away_pose, env_ids=env_id, object_ids=object_index.reshape(1))
            reward[env_id] +=1
    
    reward[terminated] = 0 # Sets terminated environments to 0
    
    return reward



def select_robot_arm_for_grasping(poses_global, robot_poses_global, origin_poses, robot_conf, device, n_obj):
   
    right_chain =   robot_conf["right_chain"]    
    left_chain =    robot_conf["left_chain"]     
    robot_chain =   robot_conf["robot_chain"]   
    left_default =  robot_conf["left_default"]   
    right_default = robot_conf["right_default"]
    origin_poses = origin_poses.squeeze(1) # (n_envs, 3)
 

    n_obj_poses = poses_global.shape[0] # Total number of object poses in all envs

    origins_expanded = origin_poses.repeat_interleave(n_obj, dim=0) # origin coordinates for each object pose as to create a common world frame
   
    obj_poses = poses_global[:,:3]- origins_expanded
    #print("Shape of obj poses in select arm: ", obj_poses.shape)
    #print("\n")
    #print("Robot pos global: ", robot_poses_global[:,:3])

    robot_poses_in_world = (robot_poses_global[:,:3]-origin_poses, robot_poses_global[:,3:7]) # Normalize robot poses with each env origin

    # Calculate end-effector poses before expanding to match object poses

    robot_default_config = assemble_full_configuration(robot_chain, (right_chain, right_default), (left_chain, left_default))

    yumiTlee, yumiTree = compute_dual_arm_end_effector_poses(robot_chain, robot_default_config) # These ones seems to be correct 
 
    wTr = pk.transform3d.Transform3d(pos=robot_poses_in_world[0], rot=robot_poses_in_world[1], device=device) # (n_envs, 4,4)

    wTlee = wTr.compose(yumiTlee.to(device)) # (n_envs, 4,4)
    wTree = wTr.compose(yumiTree.to(device)) 

    ee_left_robot = wTlee.get_matrix()[:,:3,3] #This needs to be done once for each environment
    ee_right_robot = wTree.get_matrix()[:,:3,3]

    #print("EE left robot: ", ee_left_robot)
    #print("EE right robot: ", ee_right_robot)
    ee_left_batch = ee_left_robot.repeat_interleave(n_obj, dim=0)
    ee_right_batch = ee_right_robot.repeat_interleave(n_obj, dim=0)

    l_dist = torch.linalg.norm(ee_left_batch - obj_poses, dim=1)
    r_dist = torch.linalg.norm(ee_right_batch - obj_poses,dim=1)
   
    # One if left, zero if right 
    l_o_r = (l_dist<r_dist).to(torch.uint8).to(device)

    return l_o_r


def get_grasp_pose(poses_global ,n_obj, origins, robot_poses, device):
        #TODO: Modify for multi object grasping

        """
        Grasp pose is exact same for lula and torch. This is not the issue for the faulty IK solution"""
        
        origins = origins.squeeze(1) # (n_envs, 3)
        origins_expanded = origins.repeat_interleave(n_obj, dim=0) # origin coordinates for each object pose as to create a common world frame

        obj_poses = poses_global[:,:3]- origins_expanded # Poses normalize to common orgin
        
        n_poses = obj_poses.shape[0]

        #print("Shape of origins expanded:", origins_expanded.shape)
        #print("Shape of poses global:", poses_global.shape)
        
        #print("Shape of obj poses:", obj_poses.shape)

        poses_world = (obj_poses  , poses_global[:,3:7]) # Poses in world frame

        grasp_offsets = None
        
        #print(obj_pose_in_world) # xyz qw qx qy qz
        

        #yumi_pose_in_world = self.yumi_body_link.get_world_poses()
        #yumi_pos, yumi_rot = yumi_pose_in_world[0][0], yumi_pose_in_world[1][0]
        
        robot_pos = robot_poses[:,:3]- origins
        robot_rot = robot_poses[:,3:7]
        
        robot_pos = robot_pos.repeat_interleave(n_obj, dim=0)
        robot_rot = robot_rot.repeat_interleave(n_obj, dim=0)
        
        grasp_offsets = 0.18
    
        # print("Object pose in world:", obj_pose_in_world)
        #print("Shape of poses world:", poses_world[0].shape)
        #print("Shape of orientations world:", poses_world[1].shape)
        grasp_pose_o_tran = [0, 0, grasp_offsets]
        grasp_pose_o_rot = Rotation.from_euler('xyz', [0, 3.14, 0]).as_quat()

        sf_grasp_o_rot = np.array([grasp_pose_o_rot[3], grasp_pose_o_rot[0],grasp_pose_o_rot[1], grasp_pose_o_rot[2]]) #Scalar first

        # Find transform from world to goal
        oTg_arr = pk.transform3d.Transform3d(rot=sf_grasp_o_rot, pos=grasp_pose_o_tran, device=device)#
        wTo_arr = pk.transform3d.Transform3d(pos=poses_world[0], rot=poses_world[1], default_batch_size=n_poses, device=device)
        wTg_arr = wTo_arr.compose(oTg_arr)
        
        # Find transform from robot to goal
        wTr_arr = pk.transform3d.Transform3d(pos=robot_pos, rot=robot_rot,device=device)
        rTw_arr = wTr_arr.inverse()
        rTg_arr = rTw_arr.compose(wTg_arr) #Pytorch IK expects the goal in robot frame.

        #print("robot frame grasp pose: ", grasp_pose)
        matrices_rTg = rTg_arr.get_matrix()
        matrices = wTg_arr.get_matrix()

        # Produces same result as Lula solution
        pos = matrices[:,:3,3]
        rot = pk.rotation_conversions.matrix_to_quaternion(matrices[:,:3,:3])
   

        grasp_poses = torch.cat([pos,rot],dim=1)
        #print("Grasp poses: ", matrices_rTg)
        # print("Grasp pose in function:", grasp_pose)
        return grasp_poses, matrices_rTg


