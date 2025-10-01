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
from shapely.geometry.polygon import LinearRing

import matplotlib.cm as cm
import matplotlib
from tqdm import tqdm
from scipy.spatial.transform import Rotation

import isaacsim.core.utils.bounds as bounds_utils

from visual_mm_planning.utils.transformation import wrap_angle

def create_polygon(corners):
    """
    Create a Shapely Polygon from a set of corners, ensuring the correct vertex order.

    Args:
        corners (np.ndarray): Array of corners with shape (4, 2).

    Returns:
        Polygon: A Shapely Polygon object.
    """
    # Ensure the corners are ordered in a clockwise manner
    polygon = Polygon(corners)
    if not polygon.is_valid:
        polygon = polygon.convex_hull  # Fix self-intersecting polygons
    return polygon


# Isaac sim doc: https://docs.isaacsim.omniverse.nvidia.com/4.5.0/py/source/extensions/isaacsim.core.utils/docs/index.html


def plot_polygons(bbox_cache):
    """
    Plot the polygons for the robot, table, and obstacles.

    Args:
        bbox_cache: Cached bounding box data for all objects in the scene.
    """
    # Initialize the plot
    fig, ax = plt.subplots(figsize=(8, 8))

    # Robot polygon
    robot_prim_path = "/World/Robot/base_link/visuals/top/mesh"
    rv = bounds_utils.compute_obb_corners(bbox_cache, prim_path=robot_prim_path)
    robot_polygon = create_polygon(rv[:, :2])

    robot_ring = LinearRing(robot_polygon.exterior.coords)
    ax.plot(*robot_ring.xy, color="blue", label="Robot")

    # Table polygon
    table_prim_path = "/World/table"
    tv = bounds_utils.compute_obb_corners(bbox_cache, prim_path=table_prim_path)
    table_polygon = create_polygon(tv[:, :2])
    table_ring = LinearRing(table_polygon.exterior.coords)
    ax.plot(*table_ring.xy, color="green", label="Table")

    # Obstacle polygons
    obstacle_polygons = []
    for i in range(0, 5):  # Assuming 5 obstacles
        obs_prim_path = f"/World/obstacle{i}"
        ov = bounds_utils.compute_obb_corners(bbox_cache, prim_path=obs_prim_path)
        obs_polygon = create_polygon(ov[:, :2])
        obstacle_polygons.append(obs_polygon)
        obs_ring = LinearRing(obs_polygon.exterior.coords)
        ax.plot(*obs_ring.xy, color="red", label=f"Obstacle {i}" if i == 0 else None)

    # Set plot properties
    ax.set_title("Polygons Visualization")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.legend()
    ax.axis("equal")

    # Show the plot
    plt.show()

def check_if_robot_is_in_collision(bbox_cache,n_obstacles):
    robot_prim_path = "/World/Robot/base_link/visuals/top/mesh"
    rv = bounds_utils.compute_obb_corners(bbox_cache, prim_path=robot_prim_path)
    robot_polygon = create_polygon([rv[0,:2], rv[2, :2], rv[4, :2], rv[6, :2]])

    table_prim_path = "/World/table"
    tv = bounds_utils.compute_obb_corners(bbox_cache, prim_path=table_prim_path)
    table_polygon = create_polygon([tv[0,:2], tv[2, :2], tv[4, :2], tv[6, :2]])

    obstacle_polygons = []
    obstacle_polygons.append(table_polygon)

    for i in range(0, n_obstacles):
        obs_prim_path = "/World/obstacle{}".format(i)
        ov = bounds_utils.compute_obb_corners(bbox_cache, prim_path=obs_prim_path)
        obs_polygon = create_polygon([ov[0,:2], ov[2, :2], ov[4, :2], ov[6, :2]])
        obstacle_polygons.append(obs_polygon)

    collision_status = False
    for obstacle_polygon in obstacle_polygons:
        if robot_polygon.intersects(obstacle_polygon) or robot_polygon.contains(obstacle_polygon) or obstacle_polygon.contains(robot_polygon):
            collision_status = True
            break
    # print(f"Collision status: {collision_status}")
    return collision_status



def check_intersections(bounding_boxes, obj_id, obstacle_ids, intersection_threshold=0.05):
    """
    Check if the bounding box of the robot (ID 0) intersects with bounding boxes of IDs 1, 3, or 4.
    If the intersection area is less than 5% of the total area of the robot's bounding box, it is ignored.

    Args:
        bounding_boxes (list of tuples): List of bounding boxes in the format
                                        (id, x_min, y_min, x_max, y_max, score).
        obj_id (int): The ID of the object to check for intersections.
        obstacle_ids (list): List of IDs to check for intersection with the robot.
        intersection_threshold (float): The threshold for the intersection area as a fraction of the robot's bounding box area.
    Returns:
        bool: True if a significant intersection is found, False otherwise.
    """

    # Extract the bounding box of the object id
    robot_box = None
    for bbox in bounding_boxes:
        if bbox[0] == obj_id:
            robot_box = box(bbox[1], bbox[2], bbox[3], bbox[4])  # Create a shapely box
            robot_area = robot_box.area  # Calculate the area of the robot's bounding box
            break

    if robot_box is None:
        # print("Robot bounding box (ID 0) not found.")
        return False

    # Check for intersections with bounding boxes of objects with IDs in obstacle_ids
    for bbox in bounding_boxes:
        if bbox[0] in obstacle_ids:
            obstacle_box = box(bbox[1], bbox[2], bbox[3], bbox[4])  # Create a shapely box
            if robot_box.intersects(obstacle_box):  # Check for intersection
                intersection_area = robot_box.intersection(obstacle_box).area  # Calculate intersection area
                if intersection_area >= intersection_threshold * robot_area:  # Check if intersection area is significant
                    # print(f"Significant intersection found with bounding box ID {bbox[0]}")
                    return True
                else:
                    # print(f"Ignored small intersection with bounding box ID {bbox[0]} (area: {intersection_area})")
                    pass

    # print("No significant intersections found.")
    return False


def check_collision_with_table(robot_pose, table_pose, tl, tw):
    """
    Check if the robot and table polygons intersect.
    Args:
        robot_pose (tuple): A tuple containing the robot's position and orientation in Isaac Sim format.
        table_pose (tuple): A tuple containing the table's position and orientation in Isaac Sim format.
        tl (float): Length of the table.
        tw (float): Width of the table.
        safe_dist_from_table (float): Safe distance from the table.
    Returns:
        bool: True if collision, False otherwise.
    """
    robot_rot_rpy = Rotation.from_quat([robot_pose[1][1], robot_pose[1][2], robot_pose[1][3], robot_pose[1][0]]).as_euler('xyz')
    table_rot_rpy = Rotation.from_quat([table_pose[1][1], table_pose[1][2], table_pose[1][3], table_pose[1][0]]).as_euler('xyz')

    t = pow(pow(tl,2) + pow(tw,2), 0.5)
    ang = math.atan(tl/tw)
    table_c1 = [table_pose[0][0] + (t*np.cos(wrap_angle(table_rot_rpy[2] + ang))), table_pose[0][1] + (t*np.sin(wrap_angle(table_rot_rpy[2] + ang))) ]
    table_c2 = [table_pose[0][0] + (t*np.cos(wrap_angle(table_rot_rpy[2] - ang))), table_pose[0][1] + (t*np.sin(wrap_angle(table_rot_rpy[2] - ang))) ]
    table_c3 = [table_pose[0][0] + (t*np.cos(wrap_angle(table_rot_rpy[2] + ang - np.pi))), table_pose[0][1] + (t*np.sin(wrap_angle(table_rot_rpy[2] + ang - np.pi)))]
    table_c4 = [table_pose[0][0] + (t*np.cos(wrap_angle(table_rot_rpy[2] - ang + np.pi))), table_pose[0][1] + (t*np.sin(wrap_angle(table_rot_rpy[2] - ang + np.pi)))]
    
    table = [table_c1, table_c2, table_c3, table_c4, table_c1]

    rl = 0.9/2
    rw = 0.9/2
    r = pow(pow(rl,2) + pow(rw,2), 0.5)
    ang = math.atan(rl/rw) 
    r_off = 0

    robot_c1 = [robot_pose[0][0] + (r*np.cos(wrap_angle(robot_rot_rpy[2] + ang + r_off))), robot_pose[0][1] + (r*np.sin(wrap_angle(robot_rot_rpy[2] + ang + r_off))) ]
    robot_c2 = [robot_pose[0][0] + (r*np.cos(wrap_angle(robot_rot_rpy[2] - ang + r_off))), robot_pose[0][1] + (r*np.sin(wrap_angle(robot_rot_rpy[2] - ang + r_off))) ]
    robot_c3 = [robot_pose[0][0] + (r*np.cos(wrap_angle(robot_rot_rpy[2] + ang - np.pi + r_off))), robot_pose[0][1] + (r*np.sin(wrap_angle(robot_rot_rpy[2] + ang - np.pi + r_off)))]
    robot_c4 = [robot_pose[0][0] + (r*np.cos(wrap_angle(robot_rot_rpy[2] - ang + np.pi + r_off))), robot_pose[0][1] + (r*np.sin(wrap_angle(robot_rot_rpy[2] - ang + np.pi + r_off)))]
    
    robot = [robot_c1, robot_c2, robot_c3, robot_c4, robot_c1]

    table_polygon = Polygon(table)
    robot_polygon = Polygon(robot)
    collision_status = table_polygon.intersects(robot_polygon) or table_polygon.contains(robot_polygon)
    
    return collision_status
