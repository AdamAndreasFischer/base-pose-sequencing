
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


    robot_conf["right_chain"]   = right_chain
    robot_conf["left_chain"]    = left_chain
    robot_conf["robot_chain"]   = robot_chain
    robot_conf["left_default"]  = left_default
    robot_conf["right_default"] = right_default

    obj_poses = env.scene["object"].data.object_link_state_w[..., :6] # C.O.M state
    
    robot_pose = robot.data.root_state_w[..., :6] # Root state, i.e ridgeback root state [20,3]
    print(robot_pose.shape)
    n_obj = obj_poses.shape[1]
    
    env_ids = torch.arange(0,obj_poses.shape[0]).unsqueeze(-1)
    origins = env.scene.env_origins[env_ids].repeat(1,n_obj,1)

    obj_poses_flattened = obj_poses.reshape(-1, obj_poses.shape[-1]) 
    
    env_ids_expanded = env_ids.repeat_interleave(n_obj).clone() # (n_envs*n_obj, )
    
    origins_expanded = origins.reshape(-1, origins.shape[-1]) # origin coordinates for each object pose as to create a common world frame 

    select_robot_arm_for_grasping(obj_poses_flattened, robot_poses_global=robot_pose )

    i=0/0


def select_robot_arm_for_grasping(poses_global, robot_poses_global, origin_poses, robot_conf, device):
    """Multi object capable"""
    right_chain =   robot_conf["right_chain"]    
    left_chain =    robot_conf["left_chain"]     
    robot_chain =   robot_conf["robot_chain"]   
    left_default =  robot_conf["left_default"]   
    right_default = robot_chain["right_default"]

    obj_poses = poses_global[:,:3]- origin_poses
    n_obj_poses = obj_poses.shape[0]
    robot_poses_in_world = (robot_poses_global[0, :3]-origin_poses[0], robot_poses_global[0,3:6])
    print(robot_poses_in_world)
    
    robot_default_config = assemble_full_configuration(robot_chain, (right_chain, right_default), (left_chain, left_default))

    yumiTlee, yumiTree = compute_dual_arm_end_effector_poses(robot_chain, robot_default_config)
    
    wTr = pk.transform3d.Transform3d(pos=robot_poses_in_world[0][0], rot=robot_poses_in_world[1][0], device=device)

    wTlee = wTr.compose(yumiTlee.to(device))
    wTree = wTr.compose(yumiTree.to(device)) # This coinceeds with the lula kinematic solvers ee_pose in world. I.e the root_link is ridgeback. 

    """
    The ee poses will not be the same in all environments as the robot moves around. We need to calculate the ee pose for all environments.
    """


    ee_left_robot = wTlee.get_matrix()[:,:3,3].squeeze() #Only one pose for each arm. No need for batch dimension
    ee_right_robot = wTree.get_matrix()[:,:3,3].squeeze()

    ee_left_batch = ee_left_robot.repeat(n_obj_poses,1)
    ee_right_batch = ee_right_robot.repeat(n_obj_poses,1)
    
    l_dist = torch.linalg.norm(ee_left_batch - obj_poses, dim=1)
    r_dist = torch.linalg.norm(ee_right_batch - obj_poses,dim=1)

    # One if left, zero if right 
    l_o_r = (l_dist<r_dist).to(torch.uint8).to(device)

    return l_o_r


def get_grasp_pose(self, poses_world,n_obj):
        #TODO: Modify for multi object grasping

        """
        Grasp pose is exact same for lula and torch. This is not the issue for the faulty IK solution"""
        
        grasp_offsets = None
        obj_pose_in_world = poses_world[0].cpu().numpy() #xyz qw qx qy qz
        #print(obj_pose_in_world) # xyz qw qx qy qz
        obj_pose_in_world = (obj_pose_in_world[:3],obj_pose_in_world[3:])

        yumi_pose_in_world = self.yumi_body_link.get_world_poses()
        yumi_pos, yumi_rot = yumi_pose_in_world[0][0], yumi_pose_in_world[1][0]

        robot_poses_ridgeback = self.robot.get_world_pose()
        #robot_pose_in_world = (robot_poses[0][0], robot_poses[1][0])
        ridgeback_pose_in_world = (robot_poses_ridgeback[0], robot_poses_ridgeback[1])
        grasp_offsets = self.cfg.grasp_poses.cube.top
    
        # print("Object pose in world:", obj_pose_in_world)

        grasp_pose_o_tran = [grasp_offsets.x_tran, grasp_offsets.y_tran, grasp_offsets.z_tran]
        grasp_pose_o_rot = Rotation.from_euler('xyz', [grasp_offsets.x_rot, grasp_offsets.y_rot, grasp_offsets.z_rot]).as_quat()

        sf_grasp_o_rot = np.array([grasp_pose_o_rot[3], grasp_pose_o_rot[0],grasp_pose_o_rot[1], grasp_pose_o_rot[2]]) #Scalar first

        # Find transform from world to goal
        oTg_arr = pk.transform3d.Transform3d(rot=sf_grasp_o_rot, pos=grasp_pose_o_tran, device=device)#
        wTo_arr = pk.transform3d.Transform3d(pos=poses_world[:,:3], rot=poses_world[:,3:], default_batch_size=n_obj, device=device)
        wTg_arr = wTo_arr.compose(oTg_arr)
        
        # Find transform from robot to goal
        wTr_arr = pk.transform3d.Transform3d(pos=yumi_pos, rot=yumi_rot,device=device)
        rTw_arr = wTr_arr.inverse()
        rTg_arr = rTw_arr.compose(wTg_arr) #Pytorch IK expects the goal in robot frame. 
        
 
        #print("robot frame grasp pose: ", grasp_pose)
        matrices_rTg = rTg_arr.get_matrix()
        matrices = wTg_arr.get_matrix()

        # Produces same result as Lula solution
        pos = matrices[:,:3,3]
        rot = pk.rotation_conversions.matrix_to_quaternion(matrices[:,:3,:3])
   

        grasp_poses = torch.cat([pos,rot],dim=1)
        
        # print("Grasp pose in function:", grasp_pose)
        return grasp_poses, matrices_rTg