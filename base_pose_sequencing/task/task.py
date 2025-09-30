# Isaac imports
from isaacsim import SimulationApp

# This enables a livestream server to connect to when running headless
CONFIG = {
   # "renderer": "Storm",
   "width": 1280,
   "height": 720,
   "window_width": 1920,
   "window_height": 1080,
   "headless": False,
   "renderer": "RayTracedLighting",
   # "display_options": 3286,  # Set display options to show default grid,
}

#CONFIG = {
#    # "renderer": "Storm",
#    "width": 320,
#    "height": 320,
#    "window_width": 640, 
#    "window_height": 480,
#    "headless": True,
#    "renderer": "RayTracedLighting",
#    # "display_options": 3286,  # Set display options to show default grid,
#}

# Start the isaacsim application
simulation_app = SimulationApp(launch_config=CONFIG)




import omni.usd
import omni
from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid, VisualCuboid
from isaacsim.core.api.objects.ground_plane import GroundPlane
from pxr import Sdf, UsdLux
from isaacsim.core.prims import XFormPrim
from isaacsim.sensors.camera import Camera
import isaacsim.core.utils.numpy.rotations as rot_utils
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.core.utils.stage import update_stage
from pxr import Usd, UsdLux, UsdPhysics
from isaacsim.core.utils.semantics import add_update_semantics

import sys
import carb
# carb.settings.get_settings().set_int("/rtx/debugMaterialType", 0)

from isaacsim.core.prims import Articulation, SingleArticulation
from isaacsim.core.utils.stage import add_reference_to_stage, get_stage_units
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.core.utils.viewports import set_camera_view
import isaacsim.core.utils.prims as prims_utils
from isaacsim.storage.native import get_assets_root_path

from isaacsim.robot_motion.motion_generation import ArticulationKinematicsSolver, LulaKinematicsSolver
from isaacsim.robot_motion.motion_generation import interface_config_loader
from isaacsim.robot_motion.motion_generation.lula import LulaTaskSpaceTrajectoryGenerator
import isaacsim.core.utils.bounds as bounds_utils


# general python libraries
import numpy as np
# import gym
# from gym import spaces
import torch
import math
import numpy as np
from scipy.integrate import odeint
from scipy.spatial.transform import Rotation
import datetime
from datetime import time
import cv2
import time
import random
from shapely.geometry import Polygon, box
import os
from copy import deepcopy, copy

# Mushroom rl imports
from mushroom_rl.core import Environment, MDPInfo
from mushroom_rl.rl_utils import spaces
from mushroom_rl.utils.angles import normalize_angle
from mushroom_rl.utils.viewer import Viewer
from mushroom_rl.rl_utils.spaces import *

# Isaac lab
# import isaaclab.sim as sim_utils
# from isaaclab.assets import ArticulationCfg, AssetBaseCfg

# Custom
from base_pose_sequencing.utils.collision import check_if_robot_is_in_collision, plot_polygons
from visual_mm_planning.utils.transformation import wrap_angle
#from visual_mm_planning.utils.isaac import isaac_pose_to_transformation_matrix, pose_to_transformation_matrix, pose_to_isaac_pose
from base_pose_sequencing.utils.isaac import isaac_pose_to_transformation_matrix, pose_to_transformation_matrix, pose_to_isaac_pose, get_visibility_attribute, show_prim, hide_prim
# enable websocket extension
# enable_extension("omni.services.streamclient.websocket")



class Task(Environment):
    """
    MM environmnt with rectangular table, object on thetable, top-down camera with orthographic view and
    mobile manipulator comprising of Clearpath Ridgeback and ABB Yumi.  
    """

    yumi_default_joint_angles = [1.5,-1.5,0.15,0.15,-0.45,-0.45,0.6,0.6,0,0,0,0,0,0]

    def __init__(
        self ,
        cfg
        ) -> None:

        """
        Constructor.
        Args:
            horizon (int, 5): horizon of the problem;
            gamma (float, .95): discount factor.
        """
        
        self.cfg = cfg
        self.no_of_obstacles = cfg.mdp.no_of_obstacles
        
        
        # MDP properties
        gamma = cfg.mdp.gamma                                
        horizon = cfg.mdp.horizon
        dt = 1/60

        observation_space = spaces.Box(
            low=0., high=255., shape=(cfg.mdp.no_of_state_channels, cfg.mdp.height, cfg.mdp.width))
        
        # # Continuous 3D Action space: [x, y, theta]
        # action_space = Box(
        #     low=np.array([0, 0, -np.pi]),  # Lower bounds for x, y, theta
        #     high=np.array([160, 160, np.pi]),  # Upper bounds for x, y, theta
        #     shape=(3,),  # Shape of the action space
        # )

        # Discrete action space
        # num_theta_classes = cfg.mdp.no_of_action_channels  
        # num_x_positions = cfg.mdp.width  
        # num_y_positions = cfg.mdp.height 
        # num_actions = num_theta_classes * num_x_positions * num_y_positions
        # action_space = spaces.Discrete(num_actions)  # Flattened discrete action space

        action_space = spaces.Box(
            low=0., high=1., shape=(1, cfg.mdp.no_of_action_channels * cfg.mdp.height * cfg.mdp.width,))


        mdp_info = MDPInfo(observation_space, action_space, gamma, horizon, dt)
        

        self.initialize_mdp(mdp_info)
        self.initialize_parameters()

        self._set_up_scene()


    def seed(self, seed):
        self._seed = seed


    def shutdown(self):
        simulation_app.close()


    def initialize_mdp(self, mdp_info):
        super().__init__(mdp_info)


    def initialize_parameters(self):
        self._state = np.zeros((self.cfg.mdp.no_of_state_channels, self.cfg.mdp.height, self.cfg.mdp.width), dtype=np.uint8)
        self._action = np.zeros((self.cfg.mdp.no_of_action_channels, self.cfg.mdp.height, self.cfg.mdp.width), dtype=np.uint8)
        self._random = True
        self._n_steps = 0
        self.current_robot_pose = np.zeros((3,), dtype=np.float32)
        self.init_robot_base_pose = None
        self.predicted_base_pose = None
        self.deltas = [0,0,0]
        self.theta_in_world = 0
        

        # Yumi default joint angles in radians
        self.yumi_default_joint_angles = [1.5,-1.5,0.15,0.15,-0.45,0.45,0.3,0.3,0,0,0,0,0,0]
        
        # Object trackers
        self.object_idxs = None # List of object idxes
        self.picked = None #Np array with booleans showing if specific object is picked up or not

    def hard_reset(self):
        # start simulation
        self.world.reset()

        # Initialization should happen only after simulation starts
        self.robot.initialize()

    
    # TODO: Ensure that here and normalization in datasets is same
    def _get_state(self):
        """
        Get the current state as an image with 4 channels (RGB + Depth).

        Returns:
            np.ndarray: Observation of shape (4, H, W).
        """
        # Get RGB image (H, W, 3)
        rgb = self.camera.get_rgba()[:, :, :3] /255.0 # Extract RGB channels from RGBA

        # Get Depth image (H, W, 1)
        # depth range is between 1.5, to 3.0m
        # depth = np.round(self.camera.get_depth()).astype(np.uint8)[:, :, np.newaxis] /255.0 # This is how it is done in torch tensor
        depth = np.round(self.camera.get_depth() * 255.0 / np.max(self.camera.get_depth())).astype(np.uint8) [:, :, np.newaxis] /255.0

        # Get the semantic segmentation mask from the camera
        segmentation_results = self.camera.get_current_frame()["semantic_segmentation"]
        segmentation_mask = segmentation_results["data"]
        segmentation_info = segmentation_results["info"]

        robot_id = -1
        id_to_labels = segmentation_info.get('idToLabels', {})
        for key, value in id_to_labels.items():
            if 'robot' in value:
                robot_id = int(key)

        # Create a binary mask for the specified object ID
        robot_mask = (segmentation_mask == robot_id).astype(np.uint8)[:, :, np.newaxis] # assuming 0 is the robot id
     
        # cv2.imwrite('/home/lakshadeep/mask.png', robot_mask * 255) 

        # state = np.concatenate((rgb, depth), axis=-1)  # Shape: (H, W, 4)
        state = np.concatenate((rgb, depth, robot_mask), axis=-1)  # Shape: (H, W, 5)

        # Transpose to match the observation space shape (4, H, W)
        state = np.transpose(state, (2, 0, 1))  # Shape: (4, H, W)

        return state
    

    def _get_robot_pose(self,H,W):
        semantic_segmentation_info = self.camera.get_current_frame()['semantic_segmentation']
        semantic_data = semantic_segmentation_info['data']
        semantic_labes = semantic_segmentation_info['info']['idToLabels']

        robot_label = int([key for key, val in semantic_labes.items() if val == {'robot': 'ridgeback_yumi'}][0])

        robot_poses_list = np.where(semantic_data==robot_label)
        robot_pose_mask = np.zeros((1,H,W))
        
        robot_pose_mask[0,int(np.mean(robot_poses_list[0])),int(np.mean(robot_poses_list[1]))] = 1
        del semantic_segmentation_info
        return robot_pose_mask

    def move_both_arms(self, angles):
        default_action = ArticulationAction(joint_positions=angles, 
                                            joint_indices=np.array([3,4,7,8,9,10,11,12,13,14,15,16,17,18]))
        self.robot.apply_action(default_action)

    
    def move_robot(self, x, y, theta, yumi_joint_angles=None):
        orientation_rpy = [0, 0, theta]
        base_q = Rotation.from_euler('xyz', orientation_rpy, degrees=False).as_quat()    

        self.current_robot_pose = [x, y, theta]     

        current_robot_pose = self.robot.get_world_pose()

        self.robot.set_world_pose(position = np.array([x, y, current_robot_pose[0][2]]),
                                      orientation = np.array([base_q[3], base_q[0], base_q[1], base_q[2]]))
        if yumi_joint_angles is not None:
            self.move_both_arms(yumi_joint_angles)
        else:
            self.move_both_arms(self.yumi_default_joint_angles)
        
        # NOTe: This is very important as actions are only applied after the world is stepped
        for j in range(self.cfg.render_steps):
            self.world.step(render=self.cfg.render)

    
    def move_object(self, x, y, theta, obj_id):
        """ 
        Moves the object(s) at reset
        x: np.array [n_obj,]
        y: np.array [n_obj,]
        theta: np.array [n_obj,]
        """
        orientation_rpy = np.zeros((len(obj_id),3),dtype=np.float32)
        orientation_rpy[:,2] = theta
        
        
        base_q = Rotation.from_euler('xyz', orientation_rpy, degrees=False).as_quat()         
        obj_poses = []
        for i in obj_id:
            pose = self.objects[i].get_world_poses()
            obj_poses.append((pose[0][0], pose[1][0]))
        dx, dy = 0,0
        for i,pose in enumerate(obj_poses):
            if self.cfg.debug:
                dx = i/10
                dy = i/10
            self.objects[i].set_world_poses(positions = np.array([[x[i]+dx, y[i]+dy, pose[0][2]]]),
                                      orientations = np.array([[base_q[i][3], base_q[i][0], base_q[i][1], base_q[i][2]]]))
        
        # NOTe: This is very important as actions are only applied after the world is stepped
        for j in range(self.cfg.render_steps):
            self.world.step(render=self.cfg.render)

    
    def reset_debug(self):

        no_obj = len(self.objects)
        obj_y = np.random.uniform(0 + self.cfg.task.obj_safety_distance, 
                                  self.cfg.mdp.table_dimensions.y_max - self.cfg.task.obj_safety_distance, (no_obj,))
        obj_x = np.random.uniform(self.cfg.mdp.table_dimensions.x_min/2 + self.cfg.task.obj_safety_distance, 
                                  self.cfg.mdp.table_dimensions.x_max/2 - self.cfg.task.obj_safety_distance, (no_obj,))
        obj_theta = np.random.uniform(-np.pi, np.pi, (no_obj,))
        obj_y[2] = -0.3
        self.reset_table_and_object()
        self.move_object(obj_x,obj_y, obj_theta, self.object_idxs)


        self.move_both_arms(self.yumi_default_joint_angles)
        self.robot.set_world_pose(self.robot_debug_pose[0],self.robot_debug_pose[1])
        while True:
            for obstacle in self.obstacles:
                # Randomize the position of the obstacle
                x = random.uniform(-2.5, 2.5) 
                y = random.uniform(-2.5, 2.5)

                # Randomize orientation along z-axis
                z_angle = random.uniform(0, 2 * np.pi)  # Random angle in radians
                q = Rotation.from_euler('z', z_angle).as_quat()  # Convert to quaternion

                obstacle.set_world_poses(positions=np.array([[x, y, 0.1]]), orientations=np.array([[q[3], q[0], q[1], q[2]]]))
            
            cache = bounds_utils.create_bbox_cache()
            collision = check_if_robot_is_in_collision(cache, self.cfg.mdp.no_of_obstacles)

            if not collision:
                break

        for j in range(100):
            self.world.step(render=self.cfg.render)
        self._state = self._get_state()
        return self._state, {}


    def reset(self, state=None):

        self.world.reset(soft=True)
        no_obj = len(self.objects)
        self.object_idxs = np.arange(no_obj)
        self.picked = np.ones((len(self.objects),1),dtype=np.uint8)
        for obj in self.objects:
            show_prim(self.stage, obj.prim_paths[0])
        if self.cfg.debug:
            self._state, dicts = self.reset_debug()
            return self._state, dicts

        obj_y = np.random.uniform(self.cfg.mdp.table_dimensions.y_min + self.cfg.task.obj_safety_distance, 
                                  self.cfg.mdp.table_dimensions.y_max - self.cfg.task.obj_safety_distance, (no_obj,))
        obj_x = np.random.uniform(self.cfg.mdp.table_dimensions.x_min + self.cfg.task.obj_safety_distance, 
                                  self.cfg.mdp.table_dimensions.x_max - self.cfg.task.obj_safety_distance, (no_obj,))
        obj_theta = np.random.uniform(-np.pi, np.pi, (no_obj,))
        
        
        self.reset_table_and_object()

        self.move_object(obj_x, obj_y, obj_theta, self.object_idxs)

        if self.cfg.randomize:
            dx = np.random.uniform(-0.5, 0.5)
            dy = np.random.uniform(-0.5, 0.5)
            # dtheta = np.random.uniform(-np.pi, np.pi)
            dtheta = np.random.choice([i * (np.pi / 4) for i in range(-4, 4)])

            self.deltas = [dx, dy, dtheta]

            new_obj_pose = self.transform_table_and_object(dx, dy, dtheta)
            
            obj_rot = new_obj_pose[3:,:].T
            obj_x = new_obj_pose[0,:]
            obj_y = new_obj_pose[1,:]
      
            obj_theta = Rotation.from_quat(obj_rot).as_euler('xyz')
     
     
        
        while True:
            for obstacle in self.obstacles:
                # Randomize the position of the obstacle
                x = random.uniform(-2.5, 2.5) 
                y = random.uniform(-2.5, 2.5)

                # Randomize orientation along z-axis
                z_angle = random.uniform(0, 2 * np.pi)  # Random angle in radians
                q = Rotation.from_euler('z', z_angle).as_quat()  # Convert to quaternion

                obstacle.set_world_poses(positions=np.array([[x, y, 0.1]]), orientations=np.array([[q[3], q[0], q[1], q[2]]]))

            r_x = random.uniform(-2.5, 2.5)
            r_y = random.uniform(-2.5, 2.5)
            r_theta = random.choice([i * (np.pi / 4) for i in range(-4, 4)])  # Ensure r_theta is a multiple of π/4
            self.move_robot(r_x, r_y, r_theta)

            cache = bounds_utils.create_bbox_cache()
            collision = check_if_robot_is_in_collision(cache,self.cfg.mdp.no_of_obstacles)

            if not collision:
                break

        for j in range(100):
            self.world.step(render=self.cfg.render)
        self._state = self._get_state()
       
        return self._state, {}
    

    def pixel_to_world(self, pix_x, pix_y, theta_in_world):
        """
        Convert pixel coordinates to 2D world coordinates (x, y).

        Args:
            pix_x (int): Pixel x-coordinate.
            pix_y (int): Pixel y-coordinate.
            theta_in_world (float): robot orientation in world.

        Returns:
            np.ndarray: 2D World coordinates [x, y].
        """
        # Get the camera intrinsic matrix
        intrinsic_matrix = self.camera.get_intrinsics_matrix()  # Shape: (3, 3)
        # print("Intrinsic matrix:", intrinsic_matrix)

        # Note: center of the robot bounding box provided by the detector is not the center of the ridgeback!

        w_y = -(pix_x - intrinsic_matrix[0, 2])/ 25.0
        w_x = -(pix_y - intrinsic_matrix[1, 2])/ 25.0 

        step_size = 0.08
        negative_theta = theta_in_world + np.pi  # Add π to reverse the direction

        w_x = w_x + step_size * np.cos(negative_theta)
        w_y = w_y + step_size * np.sin(negative_theta)

        return np.array([w_x, w_y]) 
    
    
    def shift_array_cyclically(self, array, target_value):
        """
        Shift the array cyclically so that the target_value becomes the first element.

        Args:
            array (np.ndarray): The input array.
            target_value (float): The value to bring to the front.

        Returns:
            np.ndarray: The modified array with the target_value as the first element.
        """
        # Find the index of the target value
        target_index = np.argmin(np.abs(array - target_value))

        # Shift the array cyclically
        shifted_array = np.roll(array, -target_index)

        return shifted_array
    

    def get_robot_angle_in_world(self, theta_class):
        current_robot_pose = self.robot.get_world_pose()
        robot_rot_in_quat = Rotation.from_quat([current_robot_pose[1][1], current_robot_pose[1][2], current_robot_pose[1][3], current_robot_pose[1][0]])
        robot_rot_in_euler = robot_rot_in_quat.as_euler('xyz')

        # classes = np.arange(-np.pi, np.pi, np.pi / 4)
        classes = np.arange(0, 2*np.pi, np.pi / 4)
        # classes = self.shift_array_cyclically(classes, robot_rot_in_euler[2])

        return wrap_angle(robot_rot_in_euler[2] + classes[theta_class])
        

    def step(self, action):
        """
        Execute a step in the environment based on the discrete action.

        Args:
            action (int): The discrete action index.

        Returns:
            state (np.ndarray): The next state.
            reward (float): The reward for the current step.
            goal_status (bool): Whether the goal is achieved.
            info (dict): Additional information.
        """ 
        # Convert the discrete action index into (theta, x, y) indices
        num_theta_classes = self.cfg.mdp.no_of_action_channels
        num_x_positions = self.cfg.mdp.width
        num_y_positions = self.cfg.mdp.height

        selected_action = np.argmax(action, axis=-1)
        # action_confidence = np.max(action)
        # print("Action confidence:", action_confidence)

        theta_class, y, x = np.unravel_index(selected_action, (num_theta_classes, num_y_positions, num_x_positions))
        theta_in_world = self.get_robot_angle_in_world(theta_class)

        self.theta_in_world = theta_in_world

        x_in_world, y_in_world = self.pixel_to_world(x, y, theta_in_world)
        
        self.predicted_base_pose = np.array([x, y])

        # print("Predicted base pose (in world):", x_in_world, y_in_world, theta_in_world)

        self.move_robot(x_in_world, y_in_world, theta_in_world)
        
        self.world.step(render=self.cfg.render)

        state = self._get_state()
        reward, goal_status = self._get_reward()
        
        # delay for video
        # time.sleep(1)
        # input("Press enter to continue")
     
        return state, reward, goal_status, {}
    
    
    def _get_grasp_pose(self, obj_id):
        #TODO: Modify for multi object grasping

        grasp_offsets = None
        obj_pose_in_world = None
        # print("Robot pose in world orignal:", self.robot.get_world_pose())
        robot_poses = self.yumi_base_link.get_world_poses()
        robot_pose_in_world = (robot_poses[0][0], robot_poses[1][0])
        # print("Robot pose in world:", robot_pose_in_world)

        grasp_offsets = self.cfg.grasp_poses.cube.top
        obj_poses = self.objects[obj_id].get_world_poses()
        obj_pose_in_world = (obj_poses[0][0], obj_poses[1][0])

        # print("Object pose in world:", obj_pose_in_world)

        grasp_pose_o_tran = [grasp_offsets.x_tran, grasp_offsets.y_tran, grasp_offsets.z_tran]
        grasp_pose_o_rot = Rotation.from_euler('xyz', [grasp_offsets.x_rot, grasp_offsets.y_rot, grasp_offsets.z_rot]).as_quat()
        grasp_pose_o = np.hstack((grasp_pose_o_tran, grasp_pose_o_rot))

        oTg = pose_to_transformation_matrix(grasp_pose_o)

        wTo = isaac_pose_to_transformation_matrix(obj_pose_in_world)
        wTg = np.matmul(wTo, oTg)

        wTr = isaac_pose_to_transformation_matrix(robot_pose_in_world)    # robot base frame
        rTw = np.linalg.inv(wTr)
        rTg = np.matmul(rTw, wTg)

        # quat = Rotation.from_matrix(rTg[:3,:3]).as_quat()
        # grasp_pose = np.array([rTg[0,3], rTg[1,3], rTg[2,3], quat[0], quat[1], quat[2], quat[3]])
        
        quat = Rotation.from_matrix(wTg[:3,:3]).as_quat()
        grasp_pose = np.array([wTg[0,3], wTg[1,3], wTg[2,3], quat[0], quat[1], quat[2], quat[3]]) 
        
        # print("Grasp pose in function:", grasp_pose)
        return grasp_pose
    


    def _get_reward(self):
        reward = 0
        goal_status = False # single step MDP
        
        # This doesnt work as its not supported by Mushroom RL
        # goal_status = np.full((1, self.cfg.mdp.no_of_action_channels * self.cfg.mdp.width * self.cfg.mdp.height), True, dtype=bool)  # Fill with True by default

        robot_pose = self.robot.get_world_pose()
        table_poses = self.table.get_world_poses()
        table_pose = (table_poses[0][0], table_poses[1][0])

        tl = self.cfg.mdp.table_dimensions.y_max + self.cfg.mdp.safe_dist_from_table
        tw = self.cfg.mdp.table_dimensions.x_max + self.cfg.mdp.safe_dist_from_table

        cache = bounds_utils.create_bbox_cache()
        collision = check_if_robot_is_in_collision(cache,self.cfg.mdp.no_of_obstacles)

        if collision:
            # plot_polygons(cache)
            # print("Collision detected")
            # input("Press enter to continue")
            reward = -100
            # goal_status = True
            return reward, goal_status
        
        # Can not deepcopy prims
        picked_objects = []
        for id,obj in enumerate(self.objects):
            grasp_pose = self._get_grasp_pose(id)
         
            #grasp_poses.append((grasp_pose, obj)) # Tuple to keep track of object and corresponding pose

            print(grasp_pose)

            arm_id = self.select_robot_arm_for_grasping(id)

            robot_base_translation,robot_base_orientation = self.robot.get_world_pose()

            target_pos = np.array([grasp_pose[0], grasp_pose[1], grasp_pose[2]])
            target_rot = np.array([grasp_pose[6], grasp_pose[3], grasp_pose[4], grasp_pose[5]])

            success = False
            if arm_id == 0:
                self.lula_kinematics_solver_left.set_robot_base_pose(robot_base_translation, robot_base_orientation)
                action, success = self.articulation_kinematics_solver_left.compute_inverse_kinematics(target_pos, target_rot)
            else:
                self.lula_kinematics_solver_right.set_robot_base_pose(robot_base_translation, robot_base_orientation)
                action, success = self.articulation_kinematics_solver_right.compute_inverse_kinematics(target_pos, target_rot)
            print(action)
            print(success)
            if success:
                self.robot.apply_action(action)
                reward += 100
                picked_objects.append(obj)
                
            else:
                reward += 0
                pass
                # print("IK did not converge to a solution.  No action is being taken")

            for j in range(self.cfg.render_steps):
                # time.sleep(0.01)
                self.world.step(render=True)
        # Turns prims invisible for next step
        for p_obj in picked_objects:
            hide_prim(self.stage, p_obj.prim_paths[0])

        return reward, goal_status
    

    def stop(self):
        # simulation_app.close()
        pass


    def reset_table_and_object(self):
        table_z_offset = 0.5
        self.table.set_world_poses(positions=np.array([[0.0, 0.0, 0.25]]), orientations=np.array([[0, 0, 0, 1]]))
        self.objects[0].set_world_poses(positions=np.array([[0.5, -0.35, table_z_offset+0.05]]), orientations=np.array([[0, 0, 0, 1]]))


    # we assume there is only one object
    def transform_table_and_object(self, dx, dy, dtheta):
        table_poses = self.table.get_world_poses()
        table_pose = (table_poses[0][0], table_poses[1][0])
        num_obj = len(self.objects)
        
        #object_poses_raw = [object.get_world_poses() for object in self.objects]
        #print("Object poses")
        #print(object_poses_raw) # each entry is [xyz] [qx qy qz qw]
#
        #object_poses = [(obj[0][0], obj[1][0]) for obj in object_poses_raw] 
        #print(object_poses)
        #object_pose = (object_poses[0][0], object_poses[1][0])
 
        object_pose = []
        # Create tuple of (pos, quat) for each object 

        for obj in self.objects:
            pose = obj.get_world_poses()
            object_pose.append((pose[0][0],pose[1][0]))
        
        wTo_arr = np.zeros((num_obj,  4, 4), dtype=np.float32)
        for i, pose in enumerate(object_pose):
            wTo_arr[i] = isaac_pose_to_transformation_matrix(pose) # [5, 4, 4]
        
        wTt = isaac_pose_to_transformation_matrix(table_pose)
        
        tTo = np.matmul(np.linalg.inv(wTt), wTo_arr) #This gives rotations only around z
        print(tTo.shape)
        # Debug
        
        ex, ey, ez = Rotation.from_quat([table_pose[1][1], table_pose[1][2], table_pose[1][3], table_pose[1][0]]).as_euler('xyz')
        ez = wrap_angle(ez + dtheta)
        t_quat = Rotation.from_euler('xyz', [ex, ey, ez]).as_quat()

        new_table_pose = np.array([table_pose[0][0] + dx, table_pose[0][1] + dy, table_pose[0][2], t_quat[0], t_quat[1], t_quat[2], t_quat[3]])
        new_table_pose = pose_to_isaac_pose(new_table_pose)
        # new_table_pose = (np.array([table_pose[0][0] + dx, table_pose[0][1] + dy, table_pose[0][2]]), table_pose[1]) # Keep the original orientation of the table

        wTti = isaac_pose_to_transformation_matrix(new_table_pose)

        wToi = np.matmul(wTti, tTo) # This also give only rotation around z axis

        print("################## DEBUG ##################")
        for i in range(num_obj):
            rot_mat = wToi[i][:3,:3]
            print(rot_mat)
            r = Rotation.from_matrix(rot_mat)
            print(r.as_euler('xyz', degrees=True))

        quats = Rotation.from_matrix(wToi[:,:3,:3]).as_quat()
        new_object_pose = np.array([wToi[:,0,3], wToi[:,1,3], wToi[:,2,3], quats[:,0], quats[:,1], quats[:,2], quats[:,3]])
        #print("New object pose")
        #print(new_object_pose.shape)
        #print(new_object_pose)
        #new_object_pose = pose_to_isaac_pose(new_object_pose)
        #print("Isaac sim new object pose")
        #print(new_object_pose)
        
        self.table.set_world_poses(positions = np.array([[new_table_pose[0][0], new_table_pose[0][1], new_table_pose[0][2]]]),
                                      orientations = np.array([[new_table_pose[1][0], new_table_pose[1][1], new_table_pose[1][2], new_table_pose[1][3]]]))
        print("########### DEBUG LOOP #################")
        for i in range(len(self.objects)):
            new_pose = new_object_pose[:,i]
            rot = Rotation.from_quat([new_pose[3], new_pose[4], new_pose[5], new_pose[6]])
            print(rot.as_euler("xyz", degrees=True))
            #print(new_pose)
            self.objects[i].set_world_poses(positions = np.array([[new_pose[0], new_pose[1], new_pose[2]]]),
                                      orientations = np.array([[new_pose[6], new_pose[3], new_pose[4], new_pose[5]]]))

        for j in range(self.cfg.render_steps):
            self.world.step(render=self.cfg.render)

        return np.array([wToi[:,0,3], wToi[:,1,3], wToi[:,2,3], quats[:,3], quats[:,0], quats[:,1], quats[:,2]])


    # You must initialize the scene before calling this function
    def apply_camera_settings(self, cam):
        # NOTE: https://docs.isaacsim.omniverse.nvidia.com/4.5.0/py/source/extensions/isaacsim.sensors.camera/docs/index.html
                
        
        # # OpenCV camera matrix and width and height of the camera sensor, from the calibration file
        # width, height = 640, 480
        # camera_matrix = [[614.0097, 0.0, 323.0128], [0.0, 614.2431, 234.18165], [0.0, 0.0, 1.0]]

        # # Pixel size in microns, aperture and focus distance from the camera sensor specification
        # # Note: to disable the depth of field effect, set the f_stop to 0.0. This is useful for debugging.
        # pixel_size = 12 * 1e-3   # in mm, 3 microns is a common pixel size for high resolution cameras
        # f_stop = 1.8            # f-number, the ratio of the lens focal length to the diameter of the entrance pupil
        # focus_distance = 3.5    # in meters, the distance from the camera to the object plane

        # # Calculate the focal length and aperture size from the camera matrix
        # ((fx,_,cx),(_,fy,cy),(_,_,_)) = camera_matrix
        # horizontal_aperture =  pixel_size * width                   # The aperture size in mm
        # vertical_aperture =  pixel_size * height
        # focal_length_x  = fx * pixel_size
        # focal_length_y  = fy * pixel_size
        # focal_length = (focal_length_x + focal_length_y) / 2         # The focal length in mm

        # # Set the camera parameters, note the unit conversion between Isaac Sim sensor and Kit
        # # cam.set_focal_length(focal_length / 10.0)                # Convert from mm to cm (or 1/10th of a world unit)
        # # cam.set_focus_distance(focus_distance)                   # The focus distance in meters
        # # cam.set_lens_aperture(f_stop * 100.0)                    # Convert the f-stop to Isaac Sim units
        # # cam.set_horizontal_aperture(horizontal_aperture / 10.0)  # Convert from mm to cm (or 1/10th of a world unit)
        # # cam.set_vertical_aperture(vertical_aperture / 10.0)

        cam.set_focal_length(5)  
        cam.set_focus_distance(3.5) 
        cam.set_horizontal_aperture(6.45)
        cam.set_vertical_aperture(6.45)

        cam.set_projection_mode("orthographic") 
        cam.set_clipping_range(0.5, 10)                      # Set the near and far clipping planes in meters


    def select_robot_arm_for_grasping(self, obj_id):
        obj_poses = self.objects[obj_id].get_world_poses()
        obj_pose_in_world = (obj_poses[0][0], obj_poses[1][0])

        robot_pose_in_world = self.robot.get_world_pose()

        wTr = isaac_pose_to_transformation_matrix(robot_pose_in_world)
        wTo = isaac_pose_to_transformation_matrix(obj_pose_in_world)

        rTo = np.matmul(np.linalg.inv(wTr), wTo)
        obj_pose_in_robot = (rTo[0,3], rTo[1,3], rTo[2,3])

        # print(self.lula_kinematics_solver_left.get_joint_names())
        # print(self.lula_kinematics_solver_right.get_joint_names())

        self.lula_kinematics_solver_left.set_robot_base_pose(robot_pose_in_world[0], robot_pose_in_world[1])
        self.lula_kinematics_solver_right.set_robot_base_pose(robot_pose_in_world[0], robot_pose_in_world[1])

        ee_left_robot, ee_rot_mat_left = self.articulation_kinematics_solver_left.compute_end_effector_pose()
        ee_right_robot, ee_rot_mat_right = self.articulation_kinematics_solver_right.compute_end_effector_pose()

        l_dist = np.linalg.norm(ee_left_robot - obj_pose_in_world[0])
        r_dist = np.linalg.norm(ee_right_robot - obj_pose_in_world[0])

        # print("Left arm distance:", l_dist)
        # print("Right arm distance:", r_dist)

        if l_dist < r_dist:
            return 0
        else:
            return 1
        

    def _set_up_scene(self) -> None:

        print("########################### Setup Scene #######################")
        # preparing the scene
        # assets_root_path = get_assets_root_path()
        # print("Isaac sim assets root path:", assets_root_path)
        # if assets_root_path is None:
        #     carb.log_error("Could not find Isaac Sim assets folder")
        #     simulation_app.close()
        #     sys.exit()

        # Add Light Source
        self.stage = omni.usd.get_context().get_stage()
        # distantLight = UsdLux.DistantLight.Define(stage, Sdf.Path("/DistantLight"))
        # distantLight.CreateIntensityAttr(300)
        # distantLight.GetShadowEnableAttr().Set(False)

        domeLight = UsdLux.DomeLight.Define(self.stage, Sdf.Path("/DomeLight"))
        domeLight.CreateIntensityAttr(2000)  # Set intensity
        # domeLight.CreateTextureFileAttr('/home/lakshadeep/abandoned_greenhouse_1k.hdr')  # Optional: Set HDRI texture file path for environment lighting

       
        # start a world to step simulator
        self.world = World(stage_units_in_meters=1.0)

        self.world.scene.add_default_ground_plane()  # add ground plane
        self.world.get_physics_context().enable_gpu_dynamics(True)

        # Add Ground Plane
        GroundPlane(prim_path="/World/GroundPlane", z_position=0.01, color=np.array([10/255.0, 10/255.0, 10/255.0]), size=100)

        # set camera view
        set_camera_view(
            eye=[5.0, 0.0, 1.5], target=[0.00, 0.00, 1.00], camera_prim_path="/OmniverseKit_Persp"
        )  

        asset_path = self.cfg.path_prefix + self.cfg.robot_usd_file
        add_reference_to_stage(usd_path=asset_path, prim_path="/World/Robot")  # add robot to stage

        self.robot = SingleArticulation(prim_path="/World/Robot", name="robot")  # create an articulation object
        UsdPhysics.ArticulationRootAPI.Apply(omni.usd.get_prim_at_path("/World/Robot"))

        if self.cfg.debug:
            quat_rot = Rotation.from_euler('xyz', [0 ,0 ,-90], degrees=True).as_quat()
            quat = [quat_rot[3], quat_rot[0], quat_rot[1], quat_rot[2]]
            self.robot.set_world_pose(position=np.array([0.0, 1.0, 0.0]) / get_stage_units(), orientation = quat)
            self.robot_debug_pose = self.robot.get_world_pose()
            print(type(self.robot))
            print("##### Seting up scene ######")
            print(omni.usd.get_prim_descendents(omni.usd.get_prim_at_path("/World/Robot")))
        else:
            self.robot.set_world_pose(position=np.array([0.0, 1.0, 0.0]) / get_stage_units())
        
        #add_update_semantics(omni.usd.get_prim_at_path("/World/Robot"), semantic_label= "robot",type_label= 'class')

        # Lula Kinematics solver
        self.lula_kinematics_solver_left = LulaKinematicsSolver(
            robot_description_path = self.cfg.path_prefix + self.cfg.motion.robot_descriptor_file_left,
            urdf_path = self.cfg.path_prefix + self.cfg.motion.robot_urdf_file
        )

        self.lula_kinematics_solver_right = LulaKinematicsSolver(
            robot_description_path = self.cfg.path_prefix + self.cfg.motion.robot_descriptor_file_right,
            urdf_path = self.cfg.path_prefix + self.cfg.motion.robot_urdf_file
        )

        self.articulation_kinematics_solver_right = ArticulationKinematicsSolver(self.robot, self.lula_kinematics_solver_right, "gripper_r_base")
        self.articulation_kinematics_solver_left = ArticulationKinematicsSolver(self.robot, self.lula_kinematics_solver_left, "gripper_l_base")
        
        # Initialize a LulaCSpaceTrajectoryGenerator object
        self.task_space_trajectory_generator = LulaTaskSpaceTrajectoryGenerator(
            robot_description_path = self.cfg.path_prefix + self.cfg.motion.robot_descriptor_file,
            urdf_path = self.cfg.path_prefix + self.cfg.motion.robot_urdf_file
        )

        # NOTE
        # VisualCuboid - no collision 
        # DynamicCuboid - collision

        # Table for objects 
        # table = VisualCuboid(
        #     prim_path="/World/table",
        #     name="table",
        #     position=np.array([0.0, 0.0, 0.25]),
        #     scale=np.array([1.6, 0.8, 0.50]),
        #     size=1.0,
        #     color=np.array([255, 255, 255]),
        # )
        # self.table = XFormPrim("/World/table", name="table")

        add_reference_to_stage(usd_path=self.cfg.path_prefix + self.cfg.table_usd_file, prim_path="/World/table")
        # add_reference_to_stage(usd_path=assets_root_path + self.cfg.table_usd_file, prim_path="/World/table")
        self.table = XFormPrim("/World/table", name="table")
        self.table.set_world_poses(positions=np.array([[0.0, 0.0, 0.25]]), orientations=np.array([[0, 0, 0, 1]]))
        # self.table.set_local_scales(np.array([[1.6, 0.8, 0.5]]))
        self.table.set_local_scales(np.array([[1, 1, 1]]))

        # For any values above 0.5 and otientation is not 0 or 90,there is problem with the rendering of the table. 
        # One possible solution could be join multiple blocks together
        # self.table.set_local_scales(np.array([[0.5, 0.5, 0.5]]))

        no_of_objects = self.cfg.mdp.no_of_objects
        table_z_offset = 0.5
        self.objects = []
        suffix = ".usd"
        for i in range(1, no_of_objects+1):
            # NOTE: cannot be used as for retrieving semantic data, labels must be added in usd file 
            # obj = VisualCuboid(
            #     prim_path="/World/obj" + str(i),
            #     name="obj" + str(i),
            #     position=np.array([0.5, -0.35, table_z_offset+0.025]),
            #     scale=np.array([0.05, 0.05, 0.05]),
            #     size=1.0,
            #     # color=np.array([random.randint(0,255), random.randint(0,255), random.randint(0,255)]),
            #     color=np.array([0, 255, 0])
            # )
            # obj = XFormPrim("/World/obj" + str(i), name="obj" + str(i))
            obst_path = self.cfg.object_usd_file + str(i)+ suffix
            
            exists = os.path.exists(self.cfg.path_prefix+obst_path)

            if not exists:
                print(f"There are only {i} unique cube usd files. Create more for more obstacles", "\n")
                print(f"Using only {i} objects")
                continue
            add_reference_to_stage(usd_path=self.cfg.path_prefix + obst_path, prim_path="/World/obj" + str(i))
            obj = XFormPrim("/World/obj" + str(i), name="obj" + str(i))
            obj.set_world_poses(positions=np.array([[0.5, -0.35, table_z_offset+0.05]]), orientations=np.array([[0, 0, 0, 1]]))
            obj.set_local_scales(np.array([[0.1, 0.1, 0.1]]))

            self.objects.append(obj)

        self.picked = np.ones((len(self.objects),1),dtype=np.uint8)

        # Yumi base link for computing grasp poses
        self.yumi_base_link = XFormPrim("/World/Robot/base_link/yumi_pedestal/yumi_base_link", name="yumi_base_link")

        # top down camera
        self.camera = Camera(
            prim_path="/World/camera",
            position=np.array([0.0, 0.0, 2.8]),
            frequency=20,
            resolution=(160, 160),
            orientation=rot_utils.euler_angles_to_quats(np.array([0, 90, 0]), degrees=True),
        )

        self.world.reset()
        self.robot.initialize()
        
        # Initialize camera
        # NOTE: https://docs.omniverse.nvidia.com/py/isaacsim/source/extensions/omni.isaac.sensor/docs/index.html
        # Semantic label must be added in the semantic schema editor (USD file inside Isaac sim)
        self.camera.initialize()
        self.apply_camera_settings(self.camera) # must be called after intiialization
        self.camera.add_distance_to_image_plane_to_frame()
        self.camera.add_bounding_box_2d_tight_to_frame()
        self.camera.add_semantic_segmentation_to_frame()


        self.obstacles = []

        # Add obstacles to the scene
        for i in range(self.no_of_obstacles):
            # Randomly select between cube or cylinder
            # shape = random.choice(["cube", "cylinder"])

            # if shape == "cube":
            #     usd_path = self.cfg.path_prefix + self.cfg.cube_obstacle_usd_file  # Path to the cube USD file
            # else:
            #     usd_path = self.cfg.path_prefix + self.cfg.cylinder_obstacle_usd_file  # Path to the cylinder USD file

            usd_path = self.cfg.path_prefix + self.cfg.cube_obstacle_usd_file  # Path to the cube USD file

            add_reference_to_stage(usd_path=usd_path, prim_path="/World/obstacle" + str(i))
            obstacle = XFormPrim("/World/obstacle" + str(i), name="obstacle" + str(i))
            
            # Randomize the position of the obstacle
            x = random.uniform(-2.5, 2.5) 
            y = random.uniform(-2.5, 2.5)

            # Randomize orientation along z-axis
            z_angle = random.uniform(0, 2 * np.pi)  # Random angle in radians
            q = Rotation.from_euler('z', z_angle).as_quat()  # Convert to quaternion

            obstacle.set_world_poses(positions=np.array([[x, y, 0.05]]), orientations=np.array([[q[3], q[0], q[1], q[2]]]))
            obstacle.set_local_scales(np.array([[0.5, 0.5, 0.5]]))

            self.obstacles.append(obstacle)


        # NOTE: not significant increase in RAM usage with below, so problem proabably lies in the mushroom rl
        # Code below can be used to test camera calibration
        
        # for i in range(0, 500):
        #     self.world.step(render=True)

        # for i in range(0, 300000):
        #     self.world.step(render=True)
        #     # self.step([np.random.randint(0, 8 * 160 * 160 + 1)])
        #     self.reset()
        #     # time.sleep(0.1)
        #     input("Press enter to continue")



        