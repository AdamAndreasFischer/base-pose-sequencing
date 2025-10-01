import pytorch_kinematics as pk
import numpy as np
from scipy.spatial.transform import Rotation
import torch

def torch_joint_pose(chain : pk.SerialChain, rob_tf: pk.Transform3d, joint_angles: torch.tensor, device: str):
    """
    Find pose of joint in torch chain
    args:
    chain: Chain object describing the robot
    robt_tf: Transform for base link in world
    joint_angles: Initial joint angles for non static joints in manipulator
    Returns: 
    trans: torch.tensor Translation of joint
    rot_mat: torch.tensor Rotation matrix of joint
    """
    _,pose = chain.jacobian(joint_angles, ret_eef_pose=True)
    pTb = pk.transform3d.Transform3d(matrix=pose, device=device)

    pTw = rob_tf.compose(pTb)

    transf_mat = pTw.get_matrix().squeeze()

    trans = transf_mat[:3,3].unsqueeze(0)
    rot_mat=transf_mat[:3,:3].unsqueeze(0)

    return trans, rot_mat