# Isaac imports
from isaacsim import SimulationApp

visaulize = False
# This enables a livestream server to connect to when running headless
if visaulize:
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
else:
    CONFIG = {
        # "renderer": "Storm",
        "width": 320,
        "height": 320,
        "window_width": 640, 
        "window_height": 480,
        "headless": True,
        "renderer": "RayTracedLighting",
        # "display_options": 3286,  # Set display options to show default grid,
    }

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

import pytorch_kinematics as pk

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
from base_pose_sequencing.utils.torch_kinematics import torch_joint_pose, compute_dual_arm_end_effector_poses, assemble_full_configuration, _summarize_results, _sample_reachable_targets, find_ee_poses
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

        print("############# pxr test ######################")
        print(("cameraProjectionType", Sdf.ValueTypeNames.Token))


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

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        torch.set_default_device(self.device)
        

        # Yumi default joint angles in radians
        #self.yumi_default_joint_angles = [1.5,-1.5,0.15,0.15,-0.45,0.45,0.3,0.3,0,0,0,0,0,0]
        self.yumi_default_joint_angles = [-1.27,1.262,-1.84,-1.84,0.28,-0.398,0.49,0.362,2.08,-2.114,1.94, 1.950,-0.03, 0.129]
        #self.yumy_default_joint_angles = [0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        self.default_tensor = torch.tensor(self.yumi_default_joint_angles, device=self.device)
        self.left_default = self.default_tensor[0::2].unsqueeze(0).to(device=self.device)
        self.right_default = self.default_tensor[1::2].unsqueeze(0).to(device=self.device)

        #self.init_config_r = torch.tensor([[1.262, -1.84, -0.398,0.362,  -2.114,  1.950, 0.129,0]],device=self.device)
        #self.init_config_l= torch.tensor([[-1.27, -1.84, 0.28, 0.49, 2.08, 1.94, -0.03,0]],device=self.device)        
        
        self.right_indices = np.array([4,  8, 10,12, 14, 16, 18])
        self.left_indices = np.array([3,  7, 9,  11, 13, 15, 17])
  
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
            #if self.cfg.debug:
            #    dx = i/10
            #    dy = i/10
            self.objects[i].set_world_poses(positions = np.array([[x[i]+dx, y[i]+dy, pose[0][2]]]),
                                      orientations = np.array([[base_q[i][3], base_q[i][0], base_q[i][1], base_q[i][2]]]))
        
        # NOTE: This is very important as actions are only applied after the world is stepped
        for j in range(self.cfg.render_steps):
            self.world.step(render=self.cfg.render)

    
    def reset_debug(self):

        no_obj = len(self.objects)
        while True:
            redo = False
            obj_y = np.random.uniform(self.cfg.mdp.table_dimensions.y_min+ self.cfg.task.obj_safety_distance, 
                                    self.cfg.mdp.table_dimensions.y_max - self.cfg.task.obj_safety_distance, (no_obj,))
            obj_x = np.random.uniform(self.cfg.mdp.table_dimensions.x_min/2 + self.cfg.task.obj_safety_distance, 
                                    self.cfg.mdp.table_dimensions.x_max/2 - self.cfg.task.obj_safety_distance, (no_obj,))
            tol = 0.1
            dist = obj_y+obj_x
            
            for i,d in enumerate(dist):
                max_dist = 0
                
                for r in dist[i+1:]:
                    diff = abs(r-d)
                    print(diff)
                    if diff>max_dist:
                        max_dist=diff
                    if diff < tol:
                        redo = True
                        continue
                continue
            if not redo:
                break


            

        #obj_y = np.array([self.cfg.mdp.table_dimensions.y_max-0.05]).repeat(no_obj)
        #obj_x = np.array([0])#[self.cfg.mdp.table_dimensions.x_max,0,self.cfg.mdp.table_dimensions.x_min])
        obj_theta = np.random.uniform(-np.pi, np.pi, (no_obj,))
        
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

        for j in range(self.cfg.render_steps):
            self.world.step(render=self.cfg.render)
        self._state = self._get_state()
        return self._state, {}


    def reset(self, state=None):

        self.world.reset(soft=True)
        no_obj = len(self.objects)
        self.object_idxs = np.arange(no_obj)
        self.picked = np.zeros((len(self.objects),1),dtype=np.uint8)
        for obj in self.objects:
            show_prim(self.stage, obj.prim_paths[0])

        if self.cfg.debug:
            self._state, dicts = self.reset_debug()
            return self._state, dicts

        while True: # Crude test to make sure that objects are distance enough from each other
            redo = False
            obj_y = np.random.uniform(self.cfg.mdp.table_dimensions.y_min+ self.cfg.task.obj_safety_distance, 
                                    self.cfg.mdp.table_dimensions.y_max - self.cfg.task.obj_safety_distance, (no_obj,))
            obj_x = np.random.uniform(self.cfg.mdp.table_dimensions.x_min/2 + self.cfg.task.obj_safety_distance, 
                                    self.cfg.mdp.table_dimensions.x_max/2 - self.cfg.task.obj_safety_distance, (no_obj,))
            tol = 0.1
            dist = obj_y+obj_x
            
            for i,d in enumerate(dist):
                max_dist = 0
                
                for r in dist[i+1:]:
                    diff = abs(r-d)
                    print(diff)
                    if diff>max_dist:
                        max_dist=diff
                    if diff < tol: #Hopefully faster then redoing all over as it is likely this will move the object enough
                        obj_y[i] = np.random.uniform(self.cfg.mdp.table_dimensions.y_min+ self.cfg.task.obj_safety_distance, 
                                    self.cfg.mdp.table_dimensions.y_max - self.cfg.task.obj_safety_distance)
                        obj_x[i] = np.random.uniform(self.cfg.mdp.table_dimensions.x_min+ self.cfg.task.obj_safety_distance, 
                                    self.cfg.mdp.table_dimensions.x_max - self.cfg.task.obj_safety_distance)
                    diff = abs(obj_y[i]+obj_x[i]-r)
                    if diff<tol:
                        redo = True
                        continue
                continue
            if not redo:
                break

        
        
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

        for j in range(self.cfg.render_steps):#self.cfg.render_steps):
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
    
    
    def _get_grasp_pose(self, poses_world,n_obj):
        #TODO: Modify for multi object grasping

        """
        Grasp pose is exact same for lula and torch. This is not the issue for the faulty IK solution"""
        
        grasp_offsets = None
        obj_pose_in_world = poses_world[0].cpu().numpy() #xyz qw qx qy qz
        #print(obj_pose_in_world) # xyz qw qx qy qz
        obj_pose_in_world = (obj_pose_in_world[:3],obj_pose_in_world[3:])

        # print("Robot pose in world orignal:", self.robot.get_world_pose())
        #robot_poses = self.yumi_base_link.get_world_poses()

        #Yumi body pose for finding object pose in robot frame for grippers. 
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
        oTg_arr = pk.transform3d.Transform3d(rot=sf_grasp_o_rot, pos=grasp_pose_o_tran, device=self.device)#
        wTo_arr = pk.transform3d.Transform3d(pos=poses_world[:,:3], rot=poses_world[:,3:], default_batch_size=n_obj, device=self.device)
        wTg_arr = wTo_arr.compose(oTg_arr)
        
        # Find transform from robot to goal
        wTr_arr = pk.transform3d.Transform3d(pos=yumi_pos, rot=yumi_rot,device=self.device)
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
    

    def _get_reward(self):
        reward = 0
        goal_status = False # single step MDP
        
        # This doesnt work as its not supported by Mushroom RL
        # goal_status = np.full((1, self.cfg.mdp.no_of_action_channels * self.cfg.mdp.width * self.cfg.mdp.height), True, dtype=bool)  # Fill with True by default

        #robot_pose = self.robot.get_world_pose()
        #table_poses = self.table.get_world_poses()
        #table_pose = (table_poses[0][0], table_poses[1][0])

        #tl = self.cfg.mdp.table_dimensions.y_max + self.cfg.mdp.safe_dist_from_table
        #tw = self.cfg.mdp.table_dimensions.x_max + self.cfg.mdp.safe_dist_from_table

        cache = bounds_utils.create_bbox_cache()
        collision = check_if_robot_is_in_collision(cache,self.cfg.mdp.no_of_obstacles)

        if collision:
            reward = -100
            goal_status = True
            return reward, goal_status
        
        # Can not deepcopy prims
        
        #no_obj = len(self.objects)
        picked_objects = []
        
        n_picked = np.count_nonzero(self.picked)
        no_obj = len(self.objects) - n_picked
        obj_order = torch.zeros((no_obj,),dtype=torch.int, device=self.device)
        poses_world =torch.zeros((no_obj,7),device=self.device)
        # Extract object poses in world frame
        j = 0

        # TODO: Make more effective if needed
        for i,obj in enumerate(self.objects):
            if self.picked[i] ==1:
                continue
            pose = obj.get_world_poses()
            trans = pose[0][0] 
            rot = pose[1][0]
            fin_pose = torch.from_numpy(np.concatenate((trans,rot)))
            poses_world[j]=fin_pose
            obj_order[j] = j
            j+=1
           
        grasp_poses, grasp_matrices = self._get_grasp_pose(poses_world, no_obj) # [N,7]
        
        arm_id = self.select_robot_arm_for_grasping(poses=poses_world, no_obj=no_obj)
        left_mask = arm_id==1
        right_mask = arm_id==0

        #robot_base_translation,robot_base_orientation = self.robot.get_world_pose()
        
        l_target_pos = grasp_matrices[left_mask]
        l_obj_id = obj_order[left_mask]
        r_target_pos = grasp_matrices[right_mask]
        r_obj_id = obj_order[right_mask]
       
        #success = torch.zeros((no_obj,),dtype=torch.uint8)
        exists_r = False
        exists_l = False
        
        #Torch IK produces low error ee-poses, however, visualization is a bit strange still. 
        if l_target_pos.shape[0]>0:
            l_targets = pk.transform3d.Transform3d(default_batch_size=l_target_pos.shape[0], matrix=l_target_pos)
            solution = self.left_kinematics_solver.solve(l_targets)
            _,left_configs = _summarize_results(self.left_chain, solution, targets=l_targets, arm_name="Left")
           
            
            action_l = left_configs
            success_l = solution.converged_any
            done_l = l_obj_id[success_l==True]
            
            if done_l.ndim ==1:
                done_l = done_l.unsqueeze(0)
            if torch.any(success_l!=False).item():
                exists_l = True
        
        if r_target_pos.shape[0]>0:
            r_targets = pk.transform3d.Transform3d(default_batch_size=r_target_pos.shape[0], matrix=r_target_pos)
            solution = self.right_kinematics_solver.solve(r_targets)
            _,right_configs = _summarize_results(self.right_chain, solution, targets=r_targets, arm_name="Right")
            
            action_r = right_configs
            success_r = solution.converged_any
            done_r = r_obj_id[success_r==True]
            
            if done_r.ndim==1:
                done_r = done_r.unsqueeze(0)
            if torch.any(success_r!=False).item():
                exists_r=True


        if exists_l and exists_r:
            picked_objects = torch.cat((done_l, done_r),dim=1).reshape(-1)
        elif exists_l: 
            picked_objects = done_l
        elif exists_r:
            picked_objects = done_r
        else:
            picked_objects = torch.tensor(())

 
        if exists_r:
            for action in action_r:
                action_nump = action.squeeze().cpu().numpy()
             
                #action = lula_joint_states
                action_articulate = ArticulationAction(action_nump,None, None, self.right_indices)
                
                self.robot.apply_action(action_articulate)
                #joint_angles = torch.stack((self.left_default.squeeze(),action),dim=1).reshape(-1).cpu().numpy()
                #self.move_both_arms(joint_angles)
                for j in range(self.cfg.render_steps):
                    # time.sleep(0.01)
                    self.world.step(render=True)
                self.move_both_arms(self.yumi_default_joint_angles)
                for j in range(self.cfg.render_steps):
                    # time.sleep(0.01)
                    self.world.step(render=True)
       
        if exists_l:
            for action in action_l:
                action_nump = action.squeeze().cpu().numpy()
                #action = lula_joint_states
                action_articulate = ArticulationAction(action_nump,None, None, self.left_indices)
             
                self.robot.apply_action(action_articulate)  
                #joint_angles = torch.stack((action,self.right_default.squeeze()),dim=1).reshape(-1).cpu().numpy()
                #self.move_both_arms(joint_angles)
                for j in range(self.cfg.render_steps):
                    # time.sleep(0.01)
                    self.world.step(render=True)
                self.move_both_arms(self.yumi_default_joint_angles)
                for j in range(self.cfg.render_steps):
                    # time.sleep(0.01)
                    self.world.step(render=True)
           
        
        if not exists_l and not exists_r:
            reward += 0
            return reward, goal_status
            # print("IK did not converge to a solution.  No action is being taken")
        
        # Turns prims invisible for next step
        if picked_objects.ndim>1:
            picked_objects = picked_objects[0]
        if not picked_objects.shape[0] == 0:
            reward += picked_objects.shape[0]*100
            for p_obj in picked_objects:
                id = p_obj.item()
                self.picked[id] =1
                hide_prim(self.stage, self.objects[id].prim_paths[0])
        
        if np.all(self.picked) == 1:
            goal_status=True
        
        if np.count_nonzero(self.picked) == len(self.objects):
            goal_status = True
        
        for j in range(self.cfg.render_steps): 
            self.world.step(render=True)
        return reward, goal_status
    

    def stop(self):
        # simulation_app.close()
        pass


    def reset_table_and_object(self):
        table_z_offset = 0.5
        self.table.set_world_poses(positions=np.array([[0.0, 0.0, 0.25]]), orientations=np.array([[0, 0, 0, 1]]))
        for obj in self.objects:
            obj.set_world_poses(positions=np.array([[0.5, -0.35, table_z_offset+0.05]]), orientations=np.array([[0, 0, 0, 1]]))


    # we assume there is only one object
    def transform_table_and_object(self, dx, dy, dtheta):
        """multi object capable"""
        table_poses = self.table.get_world_poses()
        table_pose = (table_poses[0][0], table_poses[1][0])
        num_obj = len(self.objects)
        
        
 
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
       
        # Debug
        
        ex, ey, ez = Rotation.from_quat([table_pose[1][1], table_pose[1][2], table_pose[1][3], table_pose[1][0]]).as_euler('xyz')
        ez = wrap_angle(ez + dtheta)
        t_quat = Rotation.from_euler('xyz', [ex, ey, ez]).as_quat()

        new_table_pose = np.array([table_pose[0][0] + dx, table_pose[0][1] + dy, table_pose[0][2], t_quat[0], t_quat[1], t_quat[2], t_quat[3]])
        new_table_pose = pose_to_isaac_pose(new_table_pose)
        # new_table_pose = (np.array([table_pose[0][0] + dx, table_pose[0][1] + dy, table_pose[0][2]]), table_pose[1]) # Keep the original orientation of the table

        wTti = isaac_pose_to_transformation_matrix(new_table_pose)

        wToi = np.matmul(wTti, tTo) # This also give only rotation around z axis


        quats = Rotation.from_matrix(wToi[:,:3,:3]).as_quat()
        new_object_pose = np.array([wToi[:,0,3], wToi[:,1,3], wToi[:,2,3], quats[:,0], quats[:,1], quats[:,2], quats[:,3]])

        
        self.table.set_world_poses(positions = np.array([[new_table_pose[0][0], new_table_pose[0][1], new_table_pose[0][2]]]),
                                      orientations = np.array([[new_table_pose[1][0], new_table_pose[1][1], new_table_pose[1][2], new_table_pose[1][3]]]))
      
        for i in range(len(self.objects)):
            new_pose = new_object_pose[:,i]
            
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


    def select_robot_arm_for_grasping(self, poses, no_obj):
        """Multi object capable"""
       
        obj_poses = poses[:,:3]
        
        robot_pose_in_world = self.robot.get_world_pose()#self.yumi_base_link.get_world_poses()
       
        robot_default_config = assemble_full_configuration(self.robot_chain, (self.right_chain, self.right_default), (self.left_chain, self.left_default))

        yumiTlee, yumiTree = compute_dual_arm_end_effector_poses(self.robot_chain, robot_default_config)

        wTr = pk.transform3d.Transform3d(pos=robot_pose_in_world[0], rot=robot_pose_in_world[1], device=self.device)

        wTlee = wTr.compose(yumiTlee.to(self.device))
        wTree = wTr.compose(yumiTree.to(self.device)) # This coinceeds with the lula kinematic solvers ee_pose in world. I.e the root_link is ridgeback. 

        ee_left_robot = wTlee.get_matrix()[:,:3,3].squeeze() #Only one pose for each arm. No need for batch dimension
        ee_right_robot = wTree.get_matrix()[:,:3,3].squeeze()

        ee_left_batch = ee_left_robot.repeat(no_obj,1)
        ee_right_batch = ee_right_robot.repeat(no_obj,1)
        
        l_dist = torch.linalg.norm(ee_left_batch - obj_poses, dim=1)
        r_dist = torch.linalg.norm(ee_right_batch - obj_poses,dim=1)

        # One if left, zero if right 
        l_o_r = (l_dist<r_dist).to(torch.uint8).to(self.device)

        return l_o_r
        

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
            self.robot.set_world_pose(position=np.array([0.0, 1.0, 0.05]) / get_stage_units(), orientation = quat)
            self.robot_debug_pose = self.robot.get_world_pose()
            print(type(self.robot))
            print("##### Seting up scene ######")
            print(omni.usd.get_prim_descendents(omni.usd.get_prim_at_path("/World/Robot")))
        else:
            self.robot.set_world_pose(position=np.array([0.0, 1.0, 0.05]) / get_stage_units())
        
        #add_update_semantics(omni.usd.get_prim_at_path("/World/Robot"), semantic_label= "robot",type_label= 'class')
        
        # Torch kinematics Solver

        # Robot descriptor chains

        self.robot_chain = pk.build_chain_from_urdf(open("/home/adamfi/codes/base-pose-sequencing/conf/motion/robot.urdf").read())
        self.robot_chain.to(device=self.device)
        self.right_chain = pk.SerialChain(self.robot_chain, end_frame_name="gripper_r_base", root_frame_name="yumi_body")
        self.right_chain.to(device= self.device)
        self.left_chain = pk.SerialChain(self.robot_chain, end_frame_name="gripper_l_base", root_frame_name="yumi_body")
        self.left_chain.to(device= self.device)
        

        # Joint limits. Identical to lula joint limits. Not the issue
        # Compared so that non fixed incides pair up with the joint names, i.e 1 2 7 3 4 5 6 
        lim_r = torch.tensor(self.right_chain.get_joint_limits(), device=self.device)
        lim_l = torch.tensor(self.left_chain.get_joint_limits(), device=self.device)


        self.right_kinematics_solver = pk.PseudoInverseIK(self.right_chain,
                                                        retry_configs=self.right_default, 
                                                        #num_retries=30,
                                                        joint_limits=lim_r.T,
                                                        max_iterations=30,
                                                        early_stopping_any_converged=True,
                                                        #early_stopping_no_improvement="all",
                                                        lr = 0.2
                                                          )
        self.left_kinematics_solver = pk.PseudoInverseIK(self.left_chain, 
                                                        retry_configs=self.left_default,
                                                        #num_retries=30,
                                                        joint_limits=lim_l.T,
                                                        max_iterations=30,
                                                        early_stopping_any_converged=True,
                                                        #early_stopping_no_improvement="all",
                                                        lr = 0.2
                                                          )
        

     

        add_reference_to_stage(usd_path=self.cfg.path_prefix + self.cfg.table_usd_file, prim_path="/World/table")
        # add_reference_to_stage(usd_path=assets_root_path + self.cfg.table_usd_file, prim_path="/World/table")
        self.table = XFormPrim("/World/table", name="table")
        self.table.set_world_poses(positions=np.array([[0.0, 0.0, 0.25]]), orientations=np.array([[0, 0, 0, 1]]))
        # self.table.set_local_scales(np.array([[1.6, 0.8, 0.5]]))
        self.table.set_local_scales(np.array([[1, 1, 1]]))

        # For any values above 0.5 and otientation is not 0 or 90,there is problem with the rendering of the table. 
        # One possible solution could be join multiple blocks together
        # self.table.set_local_scales(np.array([[0.5, 0.5, 0.5]]))
        self.object_dict = {}
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
                print(f"There are only {i-1} unique cube usd files. Create more for more obstacles", "\n")
                print(f"Using only {i-1} objects")
                break
            add_reference_to_stage(usd_path=self.cfg.path_prefix + obst_path, prim_path="/World/obj" + str(i))
            obj = XFormPrim("/World/obj" + str(i), name="obj" + str(i))
            self.object_dict[f"{i-1}"] = obj.name
            obj.set_world_poses(positions=np.array([[0.5, -0.35, table_z_offset+0.05]]), orientations=np.array([[0, 0, 0, 1]]))
            obj.set_local_scales(np.array([[0.1, 0.1, 0.1]]))

            self.objects.append(obj)
        print(f"Using {len(self.objects)} objects!")
        self.picked = np.zeros((len(self.objects),1),dtype=np.uint8)

        # Yumi base link for computing grasp poses
        self.yumi_base_link = XFormPrim("/World/Robot/base_link/yumi_pedestal/yumi_base_link", name="yumi_base_link")
        self.yumi_body_link = XFormPrim("/World/Robot/base_link/yumi_pedestal/yumi_base_link/yumi_body", name="yumi_body")
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



        