import numpy as np
from matplotlib import pyplot as plt

import torch

from scipy.spatial.transform import Rotation
from typing import Union
from pxr import Sdf, Usd, UsdGeom

from collections.abc import Sequence
import isaacsim.core.utils.prims as prim_utils
import isaaclab.sim as sim_utils

from pxr import Gf, Sdf


import isaaclab.sim as sim_utils

from isaaclab.assets import (

    Articulation,

    ArticulationCfg,

    AssetBaseCfg,

    RigidObject,

    RigidObjectCfg,

    RigidObjectCollection,

    RigidObjectCollectionCfg,

)

from isaaclab.scene import InteractiveScene, InteractiveSceneCfg

from isaaclab.sim import SimulationContext

from isaaclab.utils import Timer, configclass

from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

def get_visibility_attribute(
    stage: Usd.Stage, prim_path: str
) -> Union[Usd.Attribute, None]:
    """Return the visibility attribute of a prim"""
    path = Sdf.Path(prim_path)
    prim = stage.GetPrimAtPath(path)
    if not prim.IsValid():
        return None
    visibility_attribute = prim.GetAttribute("visibility")
    return visibility_attribute

def hide_prim(stage: Usd.Stage, prim_path: str):
    """Hide a prim

    Args:
        stage (Usd.Stage, required): The USD Stage
        prim_path (str, required): The prim path of the prim to hide
    """
    visibility_attribute = get_visibility_attribute(stage, prim_path)
    if visibility_attribute is None:
        return
    visibility_attribute.Set("invisible")


def show_prim(stage: Usd.Stage, prim_path: str):
    """Show a prim

    Args:
        stage (Usd.Stage, required): The USD Stage
        prim_path (str, required): The prim path of the prim to show
    """
    visibility_attribute = get_visibility_attribute(stage, prim_path)
    if visibility_attribute is None:
        return
    visibility_attribute.Set("inherited")


def isaac_pose_to_transformation_matrix(pose):
    r = Rotation.from_quat([pose[1][1], pose[1][2], pose[1][3], pose[1][0]])
    T = np.identity(4)
    T[:3,:3] = r.as_matrix()
    T[0,3] = pose[0][0] 
    T[1,3] = pose[0][1] 
    T[2,3] = pose[0][2]
    return T


def pose_to_transformation_matrix(pose):
    r = Rotation.from_quat([pose[3], pose[4], pose[5], pose[6]])
    T = np.identity(4)
    T[:3,:3] = r.as_matrix()
    T[0,3] = pose[0] 
    T[1,3] = pose[1] 
    T[2,3] = pose[2]
    return T


def pose_to_isaac_pose(pose):
    return (np.array([pose[0], pose[1], pose[2]]), np.array([pose[6], pose[3], pose[4], pose[5]]))


def isaac_pose_to_pose(p):
    return np.array([p[0][0], p[0][1], p[0][2], p[1][1], p[1][2], p[1][3], p[1][0]])


def get_transformation_matrix(pose_s, pose_d):
    wTs = isaac_pose_to_transformation_matrix(pose_s)
    wTd = isaac_pose_to_transformation_matrix(pose_d)
    sTw = np.linalg.inv(wTs)
    sTd = np.matmul(sTw, wTd)
    return sTd


def generate_object_collection(num_obj, root_path,path):
    ridgid_objects= {}

    for i in range(num_obj):

        name = "object_"+str(i)
        obj = RigidObjectCfg(
            prim_path = f"{path}/Object_{i}",
            spawn=sim_utils.UsdFileCfg(
                usd_path=root_path + "base-pose-sequencing/assets/cube.usd",
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=False, rigid_body_enabled=True),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.0),
                collision_props=sim_utils.CollisionPropertiesCfg(),
            ),
            init_state = RigidObjectCfg.InitialStateCfg(
                pos=(0.75, 0.6-i/10, 0.525), #TEST remove -i later
                rot=(0, 0, 0, 1),
            ))
        ridgid_objects[name] = obj
    return ridgid_objects


def set_visibility_multi_object(visible: bool, prim_paths:str|list|None= None,  env_ids: Sequence[int] | None = None):
    """This does not work well. Change to moving the obstacles far away instead"""
    if visible:
        attribute = "inherited"
    else:
        attribute = "invisible"

    if isinstance(prim_paths, list):
        # iterate over the environment ids
        for prim_path in prim_paths:
            prim = sim_utils.find_matching_prims(prim_path)[0]
            prim_utils.set_prim_visibility(prim, visible)
    else:
        prim = sim_utils.find_matching_prims(prim_paths)
        
        prim_utils.set_prim_visibility(prim[0], visible)

def move_object_out_of_view(object: RigidObject|None = None, env_ids: Sequence[int]|None = None):
    """Moves object out of view as visibility change is not supported on GPU physix"""



    dump = torch.tensor([99.0,99.0,1.0]) # Dump unwanted objects in a corner far away

