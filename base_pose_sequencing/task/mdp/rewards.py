
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

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def collision(env: "ManagerBasedRLEnv",
              robot: SceneEntityCfg= SceneEntityCfg("robot")):
    rev = torch.tensor([1]).to(env.device)
    return rev

#def _get_grasp_pose(poses_world,n_obj):
#    #TODO: Modify for multi object grasping
#
#    """
#    Grasp pose is exact same for lula and torch. This is not the issue for the faulty IK solution"""
#    
#    grasp_offsets = None
#    obj_pose_in_world = poses_world[0].cpu().numpy() #xyz qw qx qy qz
#    #print(obj_pose_in_world) # xyz qw qx qy qz
#    obj_pose_in_world = (obj_pose_in_world[:3],obj_pose_in_world[3:])
#
#    # print("Robot pose in world orignal:", self.robot.get_world_pose())
#    #robot_poses = self.yumi_base_link.get_world_poses()
#
#    #Yumi body pose for finding object pose in robot frame for grippers. 
#    yumi_pose_in_world = self.yumi_body_link.get_world_poses()
#    yumi_pos, yumi_rot = yumi_pose_in_world[0][0], yumi_pose_in_world[1][0]
#
#    robot_poses_ridgeback = self.robot.get_world_pose()
#    #robot_pose_in_world = (robot_poses[0][0], robot_poses[1][0])
#    ridgeback_pose_in_world = (robot_poses_ridgeback[0], robot_poses_ridgeback[1])
#    grasp_offsets = self.cfg.grasp_poses.cube.top
#
#    # print("Object pose in world:", obj_pose_in_world)
#
#    grasp_pose_o_tran = [grasp_offsets.x_tran, grasp_offsets.y_tran, grasp_offsets.z_tran]
#    grasp_pose_o_rot = Rotation.from_euler('xyz', [grasp_offsets.x_rot, grasp_offsets.y_rot, grasp_offsets.z_rot]).as_quat()
#
#    sf_grasp_o_rot = np.array([grasp_pose_o_rot[3], grasp_pose_o_rot[0],grasp_pose_o_rot[1], grasp_pose_o_rot[2]]) #Scalar first
#
#    # Find transform from world to goal
#    oTg_arr = pk.transform3d.Transform3d(rot=sf_grasp_o_rot, pos=grasp_pose_o_tran, device=self.device)#
#    wTo_arr = pk.transform3d.Transform3d(pos=poses_world[:,:3], rot=poses_world[:,3:], default_batch_size=n_obj, device=self.device)
#    wTg_arr = wTo_arr.compose(oTg_arr)
#    
#    # Find transform from robot to goal
#    wTr_arr = pk.transform3d.Transform3d(pos=yumi_pos, rot=yumi_rot,device=self.device)
#    rTw_arr = wTr_arr.inverse()
#    rTg_arr = rTw_arr.compose(wTg_arr) #Pytorch IK expects the goal in robot frame. 
#    
#
#    #print("robot frame grasp pose: ", grasp_pose)
#    matrices_rTg = rTg_arr.get_matrix()
#    matrices = wTg_arr.get_matrix()
#
#    # Produces same result as Lula solution
#    pos = matrices[:,:3,3]
#    rot = pk.rotation_conversions.matrix_to_quaternion(matrices[:,:3,:3])
#
#
#    grasp_poses = torch.cat([pos,rot],dim=1)
#    
#    # print("Grasp pose in function:", grasp_pose)
#    return grasp_poses, matrices_rTg
#
#
#def _get_reward(env: "ManagerBasedRLEnv",
#                kinematic_solver: pk.PseudoInverseIK, 
#                ):
#    reward = 0
#    goal_status = False # single step MDP
#    
#    # This doesnt work as its not supported by Mushroom RL
#    # goal_status = np.full((1, self.cfg.mdp.no_of_action_channels * self.cfg.mdp.width * self.cfg.mdp.height), True, dtype=bool)  # Fill with True by default
#
#    #robot_pose = self.robot.get_world_pose()
#    #table_poses = self.table.get_world_poses()
#    #table_pose = (table_poses[0][0], table_poses[1][0])
#
#    #tl = self.cfg.mdp.table_dimensions.y_max + self.cfg.mdp.safe_dist_from_table
#    #tw = self.cfg.mdp.table_dimensions.x_max + self.cfg.mdp.safe_dist_from_table
#
#    cache = bounds_utils.create_bbox_cache()
#    collision = check_if_robot_is_in_collision(cache,self.cfg.mdp.no_of_obstacles)
#
#    if collision:
#        reward = -100
#        goal_status = True
#        return reward, goal_status
#    
#    # Can not deepcopy prims
#    
#    #no_obj = len(self.objects)
#    picked_objects = []
#    
#    n_picked = np.count_nonzero(self.picked)
#    no_obj = len(self.objects) - n_picked
#    obj_order = torch.zeros((no_obj,),dtype=torch.int, device=self.device)
#    poses_world =torch.zeros((no_obj,7),device=self.device)
#    # Extract object poses in world frame
#    j = 0
#
#    # TODO: Make more effective if needed
#    for i,obj in enumerate(self.objects):
#        if self.picked[i] ==1:
#            continue
#        pose = obj.get_world_poses()
#        trans = pose[0][0] 
#        rot = pose[1][0]
#        fin_pose = torch.from_numpy(np.concatenate((trans,rot)))
#        poses_world[j]=fin_pose
#        obj_order[j] = j
#        j+=1
#        
#    grasp_poses, grasp_matrices = self._get_grasp_pose(poses_world, no_obj) # [N,7]
#    
#    arm_id = self.select_robot_arm_for_grasping(poses=poses_world, no_obj=no_obj)
#    left_mask = arm_id==1
#    right_mask = arm_id==0
#
#    #robot_base_translation,robot_base_orientation = self.robot.get_world_pose()
#    
#    l_target_pos = grasp_matrices[left_mask]
#    l_obj_id = obj_order[left_mask]
#    r_target_pos = grasp_matrices[right_mask]
#    r_obj_id = obj_order[right_mask]
#    
#    #success = torch.zeros((no_obj,),dtype=torch.uint8)
#    exists_r = False
#    exists_l = False
#    
#    #Torch IK produces low error ee-poses, however, visualization is a bit strange still. 
#    if l_target_pos.shape[0]>0:
#        l_targets = pk.transform3d.Transform3d(default_batch_size=l_target_pos.shape[0], matrix=l_target_pos)
#        solution = self.left_kinematics_solver.solve(l_targets)
#        _,left_configs = _summarize_results(self.left_chain, solution, targets=l_targets, arm_name="Left")
#        
#        
#        action_l = left_configs
#        success_l = solution.converged_any
#        done_l = l_obj_id[success_l==True]
#        
#        if done_l.ndim ==1:
#            done_l = done_l.unsqueeze(0)
#        if torch.any(success_l!=False).item():
#            exists_l = True
#    
#    if r_target_pos.shape[0]>0:
#        r_targets = pk.transform3d.Transform3d(default_batch_size=r_target_pos.shape[0], matrix=r_target_pos)
#        solution = self.right_kinematics_solver.solve(r_targets)
#        _,right_configs = _summarize_results(self.right_chain, solution, targets=r_targets, arm_name="Right")
#        
#        action_r = right_configs
#        success_r = solution.converged_any
#        done_r = r_obj_id[success_r==True]
#        
#        if done_r.ndim==1:
#            done_r = done_r.unsqueeze(0)
#        if torch.any(success_r!=False).item():
#            exists_r=True
#
#
#    if exists_l and exists_r:
#        picked_objects = torch.cat((done_l, done_r),dim=1).reshape(-1)
#    elif exists_l: 
#        picked_objects = done_l
#    elif exists_r:
#        picked_objects = done_r
#    else:
#        picked_objects = torch.tensor(())
#
#
#    if exists_r:
#        for action in action_r:
#            action_nump = action.squeeze().cpu().numpy()
#            
#            #action = lula_joint_states
#            action_articulate = ArticulationAction(action_nump,None, None, self.right_indices)
#            
#            self.robot.apply_action(action_articulate)
#            #joint_angles = torch.stack((self.left_default.squeeze(),action),dim=1).reshape(-1).cpu().numpy()
#            #self.move_both_arms(joint_angles)
#            for j in range(self.cfg.render_steps):
#                # time.sleep(0.01)
#                self.world.step(render=True)
#            self.move_both_arms(self.yumi_default_joint_angles)
#            for j in range(self.cfg.render_steps):
#                # time.sleep(0.01)
#                self.world.step(render=True)
#    
#    if exists_l:
#        for action in action_l:
#            action_nump = action.squeeze().cpu().numpy()
#            #action = lula_joint_states
#            action_articulate = ArticulationAction(action_nump,None, None, self.left_indices)
#            
#            self.robot.apply_action(action_articulate)  
#            #joint_angles = torch.stack((action,self.right_default.squeeze()),dim=1).reshape(-1).cpu().numpy()
#            #self.move_both_arms(joint_angles)
#            for j in range(self.cfg.render_steps):
#                # time.sleep(0.01)
#                self.world.step(render=True)
#            self.move_both_arms(self.yumi_default_joint_angles)
#            for j in range(self.cfg.render_steps):
#                # time.sleep(0.01)
#                self.world.step(render=True)
#        
#    
#    if not exists_l and not exists_r:
#        reward += 0
#        return reward, goal_status
#        # print("IK did not converge to a solution.  No action is being taken")
#    
#    # Turns prims invisible for next step
#    if picked_objects.ndim>1:
#        picked_objects = picked_objects[0]
#    if not picked_objects.shape[0] == 0:
#        reward += picked_objects.shape[0]*100
#        for p_obj in picked_objects:
#            id = p_obj.item()
#            self.picked[id] =1
#            hide_prim(self.stage, self.objects[id].prim_paths[0])
#    
#    if np.all(self.picked) == 1:
#        goal_status=True
#    
#    if np.count_nonzero(self.picked) == len(self.objects):
#        goal_status = True
#    
#    for j in range(self.cfg.render_steps): 
#        self.world.step(render=True)
#    return reward, goal_status