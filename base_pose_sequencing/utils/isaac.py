import numpy as np
from matplotlib import pyplot as plt
import hydra
from omegaconf import DictConfig, OmegaConf
import os
from shapely.geometry import Polygon
import math
import networkx as nx

from math import pi
from shapely.ops import linemerge, unary_union, polygonize
from shapely.geometry import LineString, Polygon, Point, box
from shapely.ops import split

import matplotlib.cm as cm
import matplotlib
from tqdm import tqdm
from scipy.spatial.transform import Rotation
from typing import Union
from pxr import Sdf, Usd, UsdGeom


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
