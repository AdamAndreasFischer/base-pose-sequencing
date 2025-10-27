
import pytorch_kinematics as pk
import torch
import numpy as np
from scipy.spatial.transform import Rotation
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers.manager_base import ManagerTermBase
from isaaclab.managers.manager_term_cfg import RewardTermCfg
from isaaclab.sensors import ContactSensor, RayCaster
from isaaclab.managers import SceneEntityCfg
import isaacsim.core.utils.bounds as bounds_utils

from base_pose_sequencing.utils.collision import check_if_robot_is_in_collision
from base_pose_sequencing.utils.torch_kinematics import _summarize_results
from base_pose_sequencing.utils.torch_kinematics import assemble_full_configuration, compute_dual_arm_end_effector_poses, _summarize_results, get_robot_IK, get_robot_chains

from base_pose_sequencing.task.mdp.terminations import collision_check


if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def collision(env: "ManagerBasedRLEnv",
              robot_cfg: SceneEntityCfg= SceneEntityCfg("robot"),
              reward_multiplier: float = 1.0):

    collisions = collision_check(env, threshold=100).to(torch.uint8)

    collisions = 0 - collisions # return -1 for collision, 0 if no collision

    return collisions* reward_multiplier

def navcost(env: "ManagerBasedRLEnv",
            ):
    
    """
    Function for calculating the navigation cost of the action. What is needed is to access the previous robot pose, the current robot pose, and with a path planner 
    decide what the cost of the base pose selection is. 

    TODO: Fix variable for initial robot pose
    Fix path planner 
    Costmap extraction with obstacles
    """



def pick_reward(env: "ManagerBasedRLEnv",
                ):
    """
    Reward for picking objects. This needs a kinematic solver which can process multiple poses at the same time, possibly torch IK solver. 
    TODO: Function for calculating grasp poses
    Function of choosing arm for picking

    """
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

    obj_poses = env.scene["object"].data.object_link_state_w[..., :7] # C.O.M state
    
    robot_pose = robot.data.root_state_w[..., :7] # Root state, i.e ridgeback root state [20,7] xyz qw qx qy qz
    print(robot_pose.shape)
    n_obj = obj_poses.shape[1]
    
    env_ids = torch.arange(0,obj_poses.shape[0]).unsqueeze(-1).to(device=env.device)

    env_ids_expanded = env_ids.repeat_interleave(n_obj, dim=0)
    print(env_ids_expanded)

    origins = env.scene.env_origins[env_ids]

    obj_poses_flattened = obj_poses.reshape(-1, obj_poses.shape[-1]) 
   
    arm_id = select_robot_arm_for_grasping(poses_global=obj_poses_flattened, robot_poses_global=robot_pose, origin_poses=origins, robot_conf=robot_conf, device=env.device, n_obj=n_obj)
    
    _, grasp_matrices = get_grasp_pose(poses_global=obj_poses_flattened, n_obj=n_obj, origins=origins, robot_poses=robot_pose, device=env.device)

    left_mask = (arm_id==1)
    right_mask = (arm_id==0)

    l_target_poses = grasp_matrices[left_mask]
    r_target_poses = grasp_matrices[right_mask]

    l_envs = env_ids_expanded[left_mask]
    r_envs = env_ids_expanded[right_mask]

    success_ids = torch.zeros(env.num_envs * n_obj, device=env.device, dtype=torch.uint8)
    exists_l = False
    exists_r= False


    if l_target_poses.shape[0]>0:
        l_targets = pk.transform3d.Transform3d(default_batch_size=l_target_poses.shape[0], matrix=l_target_poses, device=env.device)

        solution = left_IK.solve(
            target_poses=l_targets
        )
        _,left_configs = _summarize_results(left_chain, solution, targets= l_targets, arm_name="left")
        action_l = left_configs
        success_l = solution.converged_any
        left_success_env_id = l_envs[success_l]


        print("Successes left: ", success_l)
        print("Success env ids left: ", left_success_env_id)
        if torch.any(success_l!=False).item():
             exists_l=True
        
    if r_target_poses.shape[0]>0:
        r_targets = pk.transform3d.Transform3d(default_batch_size=r_target_poses.shape[0], matrix=r_target_poses, device=env.device)
        solution = right_IK.solve(target_poses=r_targets)
        _,right_configs = _summarize_results(right_chain, solution, targets=r_targets, arm_name="right")
        action_r = right_configs
        success_r = solution.converged_any
        right_success_env_id = r_envs[success_r]
        print("Successes right: ", success_r)
        print("Success env ids right: ", right_success_env_id)
        if torch.any(success_r!=False).item():
             exists_r=True
 
    if exists_r: # Batch joint angles
        velocities = torch.zeros_like(action_r[0])
        for i,action in enumerate(action_r):
            env_id = right_success_env_id[i]
            print("Right env id: ", env_id)
            env.scene["robot"].write_joint_state_to_sim(position= action, velocity = velocities, joint_ids= right_indices, env_ids = env_id.unsqueeze(0))
            for i in range(10):
                env.sim.render()
    if exists_l:
        velocities = torch.zeros_like(action_l[0])
        for i,action in enumerate(action_l):
            env_id = left_success_env_id[i]
            print("Left env id: ", env_id)
            env.scene["robot"].write_joint_state_to_sim(position= action, velocity = velocities ,joint_ids= left_indices, env_ids=env_id.unsqueeze(0))
            for i in range(10):
                env.sim.render()
         

    return torch.zeros(env.num_envs, device=env.device)



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