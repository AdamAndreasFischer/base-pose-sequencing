import argparse

from isaaclab.app import AppLauncher


import math
import torch
import numpy as np
import os
from pxr import UsdGeom, Gf
import omni.usd

import pytorch_kinematics as pk

import carb
carb.settings.get_settings().set("persistent/app/viewport/displayOptions", 0)

from dataclasses import MISSING
from typing import TYPE_CHECKING

import isaaclab.envs.mdp as mdp
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm

from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObject, RigidObjectCfg

from isaaclab.sensors import ContactSensorCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.camera.tiled_camera import TiledCamera
from isaaclab.sensors.camera.tiled_camera_cfg import TiledCameraCfg
from isaaclab.sensors.camera.camera_cfg import CameraCfg
from isaaclab.actuators import ImplicitActuatorCfg

from isaaclab.utils import configclass
import isaacsim.core.utils.numpy.rotations as rot_utils
import isaacsim.core.utils.prims as prim_utils
import isaaclab.sim as sim_utils



from base_pose_sequencing.task.mdp.terminations import collision_check
#from base_pose_sequencing.task.mdp.observations import obj_pose_in_robot_frame
from base_pose_sequencing.task.mdp.actions import MoveBaseActionCfg
from base_pose_sequencing.task.mdp.rewards import collision

ROOT_PATH = "/home/adamfi/codes/"



@configclass
class BasePosePlanningSceneCfg(InteractiveSceneCfg):
    """Configuration for obj tracking state scene."""


    ee_frame = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/gripper_l_finger_l",
        target_frames=[
            FrameTransformerCfg.FrameCfg(prim_path="{ENV_REGEX_NS}/Robot/gripper_l_finger_l")
        ]
    ) 

    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0), color=(10/255.0, 10/255.0, 10/255.0)),
    )

    # lights - can be disabled as IsaacLab also supports global illumination
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.99, 0.99, 0.99), intensity=2000.0),
    )


    # robot
    robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=ROOT_PATH + "ridgeback_yumi/ridgeback_yumi.usd",
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(enabled_self_collisions=True),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                # ridgeback
                "front_left_wheel": 0.0,
                "front_right_wheel": 0.0,
                "rear_left_wheel": 0.0,
                "rear_right_wheel": 0.0,
                # yumi left arm
                "yumi_joint_1_l": 1.5,
                "yumi_joint_2_l": 0.15,
                "yumi_joint_3_l": -0.45,
                "yumi_joint_4_l": 0.3,
                "yumi_joint_5_l": 0.0,
                "yumi_joint_6_l": 0.0,
                "yumi_joint_7_l": 0.0,
                # yumi left hand
                "gripper_l_joint": 0.015,
                "gripper_l_joint_m": 0.015,
                # yumi right arm
                "yumi_joint_1_r": -1.5,
                "yumi_joint_2_r": 0.15,
                "yumi_joint_3_r": 0.45,
                "yumi_joint_4_r": 0.3,
                "yumi_joint_5_r": 0.0,
                "yumi_joint_6_r": 0.0,
                "yumi_joint_7_r": 0.0,
                # yumi right hand
                "gripper_r_joint": 0.015,
                "gripper_r_joint_m": 0.015,
            },
            joint_vel={".*": 0.0}, 
            pos=(0.5, -0.75, 0.0 + 0.25), 
            rot=(1, 0, 0, 0.)
        ),
        actuators={
            "base": ImplicitActuatorCfg(
                joint_names_expr=["front_left_wheel", "rear_left_wheel", "front_right_wheel", "rear_right_wheel"],
                velocity_limit=100.0,
                effort_limit=1000.0,
                stiffness=0.0,
                damping=1e5,
            ),
            "yumi_l_shoulder": ImplicitActuatorCfg(
                joint_names_expr=["yumi_joint_1_l", "yumi_joint_2_l", "yumi_joint_3_l", "yumi_joint_4_l"],
                effort_limit=87.0,
                velocity_limit=100.0,
                stiffness=800.0,
                damping=40.0,
            ),
            "yumi_l_forearm": ImplicitActuatorCfg(
                joint_names_expr=["yumi_joint_5_l", "yumi_joint_6_l", "yumi_joint_7_l"],
                effort_limit=12.0,
                velocity_limit=100.0,
                stiffness=800.0,
                damping=40.0,
            ),
            "yumi_l_hand": ImplicitActuatorCfg(
                joint_names_expr=["gripper_l_joint", "gripper_l_joint_m"],
                effort_limit=200.0,
                velocity_limit=0.2,
                stiffness=1e5,
                damping=1e3,
            ),
            "yumi_r_shoulder": ImplicitActuatorCfg(
                joint_names_expr=["yumi_joint_1_r", "yumi_joint_2_r", "yumi_joint_3_r", "yumi_joint_4_r"],
                effort_limit=87.0,
                velocity_limit=100.0,
                stiffness=800.0,
                damping=40.0,
            ),
            "yumi_r_forearm": ImplicitActuatorCfg(
                joint_names_expr=["yumi_joint_5_r", "yumi_joint_6_r", "yumi_joint_7_r"],
                effort_limit=12.0,
                velocity_limit=100.0,
                stiffness=800.0,
                damping=40.0,
            ),
            "yumi_r_hand": ImplicitActuatorCfg(
                joint_names_expr=["gripper_r_joint", "gripper_r_joint_m"],
                effort_limit=200.0,
                velocity_limit=0.2,
                stiffness=1e5,
                damping=1e3,
            ),
        },
    )


    # Table
    table = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/table",
        spawn=sim_utils.UsdFileCfg(
            usd_path=ROOT_PATH + "base-pose-sequencing/assets/table_wood.usd",
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=100.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.5, 0.9, 0.0 + 0.25), rot=(0, 0, 0, 1)),
    )

    #device = "cuda" if torch.cuda.is_available() else "cpu"
#
    #robot_chain = pk.build_chain_from_urdf(open(ROOT_PATH+"/base-pose-sequencing/conf/motion/robot.urdf").read())
    #robot_chain.to(device= device)
    #right_chain = pk.SerialChain(robot_chain, end_frame_name="gripper_r_base", root_frame_name="yumi_body")
    #right_chain.to(device= device)
    #left_chain = pk.SerialChain(robot_chain, end_frame_name="gripper_l_base", root_frame_name="yumi_body")
    #left_chain.to(device= device)
    #print("Torch ik initated")
    # Object
    object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/object",
        spawn=sim_utils.UsdFileCfg(
            usd_path=ROOT_PATH + "base-pose-sequencing/assets/cube1.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=False, rigid_body_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.75, 0.6, 0.525), 
            rot=(0, 0, 0, 1),
        )
    )
    print("Object initiated")
    # Camera
    camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/camera",  # <-- attach relative to base_link of ridgeback
        data_types=["rgb", "depth"],
        height=128,
        width=128,
        spawn=sim_utils.OrthographicCameraCfg(
            focal_length=5.0, 
            focus_distance=3.5, 
            horizontal_aperture=6.45, 
            clipping_range=(0.5, 10.0),
             # Changed in file sensors_cfg.py and sensors.py
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0, 0.0, 2.8),          # offset relative to base_link frame
            # rot=(0.5, -0.5, 0.5, -0.5),   # w x y z: -90, 90, 0   (facing forward)    
            rot=rot_utils.euler_angles_to_quats(np.array([0, 90, 0]), degrees=True), # w x y z: -180, 75, 90 (tilted down a bit) 
            convention="ros"            # or "world/opengl/ros" if you want local frame offset
        ),
    )
    print("Camera Initiated")

    ## Contact sensors
    ## https://docs.isaacsim.omniverse.nvidia.com/4.5.0/sensors/isaacsim_sensors_physics_contact.html
    #contact_forces_table = ContactSensorCfg(
    #    prim_path="{ENV_REGEX_NS}/table/_36_wood_block",
    #    update_period=0.0,
    #    history_length=1,
    #    debug_vis=True,
    #    filter_prim_paths_expr=["{ENV_REGEX_NS}/Robot"],
    #)

@configclass
class EventCfg:
    """Configuration for events."""


    reset_robot_pose = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            # "pose_range": {'x':(-0.25, 0.25), 'y':(-2.0, -1.5), 'z':(-0., 0.), 'roll':(0., 0.), 'pitch':(0., 0.), 'yaw':(1.4, 1.6)},
            "pose_range": {'x':(0.4, 0.6), 'y':(-2.0, -1.5), 'z':(-0., 0.), 'roll':(0., 0.), 'pitch':(0., 0.), 'yaw':(1.42, 1.72)},
            "velocity_range": {'x':(-0., .0), 'y':(-0., 0.), 'z':(-0., 0.), 'roll':(0., 0.), 'pitch':(0., 0.), 'yaw':(0, 0)},
        },
    )

    reset_yumi_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["yumi_joint_1_l", "yumi_joint_2_l",
                "yumi_joint_3_l", "yumi_joint_4_l", "yumi_joint_5_l", "yumi_joint_6_l", "yumi_joint_7_l", 
                "gripper_l_joint", "gripper_l_joint_m", "yumi_joint_1_r", "yumi_joint_2_r", "yumi_joint_3_r", 
                "yumi_joint_4_r", "yumi_joint_5_r", "yumi_joint_6_r", "yumi_joint_7_r", "gripper_r_joint", 
                "gripper_r_joint_m"]),
            "position_range": (-0.0, 0.0),
            "velocity_range": (-0.0, 0.0),
        },
    )

    reset_obj_pose = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "pose_range": {'x':(-0.1, .1), 'y':(-0.1, 0.1), 'z':(-0., 0.), 'roll':(0., 0.), 'pitch':(0., 0.), 'yaw':(0, 0)},
            "velocity_range": {'x':(-0., .0), 'y':(-0., 0.), 'z':(-0., 0.), 'roll':(0., 0.), 'pitch':(0., 0.), 'yaw':(0, 0)},
        },
    )

    reset_table_pose = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("table"),
            "pose_range": {'x':(-0., .0), 'y':(-0., 0.), 'z':(-0., 0.), 'roll':(0., 0.), 'pitch':(0., 0.), 'yaw':(0, 0)},
            "velocity_range": {'x':(-0., .0), 'y':(-0., 0.), 'z':(-0., 0.), 'roll':(0., 0.), 'pitch':(0., 0.), 'yaw':(0, 0)},
        },
    )

@configclass
class ActionsCfg:
    """
    Instant move base action, i.e teleporting the base to the pose instead of "driving" it there. 
    """
    base_pose = MoveBaseActionCfg(
        asset_name = "robot",
        scale=1.0,
        clip = dict(
            x=(-2.5, 2.5),
            y=(-2.5, 2.5),
            theta=(-np.pi, np.pi)
        )
    )

@configclass
class ObservationCfg:
    """Observation class defining the observations extracted from the environment"""

    @configclass
    class PolicyCfg(ObsGroup):
        image = ObsTerm(
            func = mdp.image,
            params={"snesor_cfg":SceneEntityCfg("camera")},
        )

    policy: PolicyCfg=PolicyCfg()

@configclass
class RewardsCfg:

    terminating = RewTerm(func = collision, 
                          params={
                              "robot_cfg": SceneEntityCfg("robot")
                          },
                          weight= 1.0,
                          )

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # (1) Time out for exceeding length of episode
    time_out_el = DoneTerm(func=mdp.time_out, time_out=True)

    # (2) Time out for collision 
    time_out_collision = DoneTerm(func=collision_check, params={"threshold": 1000})


@configclass
class BasePosePlanningEnvCfg(ManagerBasedRLEnvCfg):
    """Base config for the Base Pose planning environmennt
    
    Set MISSING for not yet defined classes, e.g BasePosePlanningSceneCfg"""
    
    root_path: str = MISSING

    scene: BasePosePlanningSceneCfg = MISSING

    actions: ActionsCfg = ActionsCfg()

    observations: ObservationCfg= ObservationCfg()
    events:EventCfg = EventCfg()

    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self) -> None:
        """Post initialization."""
        print("In post")
        # general settings
        self.decimation = 2
        self.episode_length_s = 1/6        # sim.dt * decimation (_s: in seconds)
        # viewer settings
        self.viewer.eye = (8.0, 0.0, 5.0)
        # simulation settings
        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation
        print("Pre scene intiation")
        self.scene = BasePosePlanningSceneCfg(num_envs=10, env_spacing=8.0)
        print("Scene intiated")