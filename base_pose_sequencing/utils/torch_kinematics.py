import pytorch_kinematics as pk
import numpy as np
from scipy.spatial.transform import Rotation
import torch
from typing import Dict, Optional, Tuple
from pytorch_kinematics.transforms import rotation_conversions as rc

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

def find_ee_poses(full_chain: pk.Chain, rob_tf: pk.Transform3d, joint_angles: torch.tensor, device:str):
    """Find ee poses in word frame"""

    frame_indices = full_chain.get_frame_indices("gripper_l_base", "gripper_r_base").to(device=device)
    joint_poses = full_chain.forward_kinematics(joint_angles, frame_indices=frame_indices)
    rTrightee = joint_poses["gripper_r_base"].to(device=device)
    rTleftee = joint_poses["gripper_l_base"].to(device=device)

    r_ee_in_world = rob_tf.compose(rTrightee).get_matrix()
    l_ee_in_world = rob_tf.compose(rTleftee).get_matrix()

    return r_ee_in_world, l_ee_in_world




def compute_dual_arm_end_effector_poses(
    full_chain: pk.Chain, joint_values: torch.Tensor | Dict[str, torch.Tensor | float | int]
) -> Tuple[pk.Transform3d, pk.Transform3d]:
    """Compute the left and right wrist poses using the complete robot chain."""

    frame_indices = full_chain.get_frame_indices("gripper_l_base", "gripper_r_base")
    fk = full_chain.forward_kinematics(joint_values, frame_indices=frame_indices)
    return fk["gripper_l_base"], fk["gripper_r_base"]




def assemble_full_configuration(
    full_chain: pk.Chain,
    partial_configs: Tuple[pk.SerialChain, torch.Tensor],
    *additional_partials: Tuple[pk.SerialChain, torch.Tensor],
) -> torch.Tensor:
    """Create a full-chain joint vector given per-arm configurations.

    Any joints not specified are filled with zeros.

    Example: 
    default_angles = torch.tensor(YUMI_DEFAULT_JOINT_ANGLES, dtype=full_chain.dtype, device=full_chain.device)
    left_default = default_angles[0::2]
    right_default = default_angles[1::2]
    default_config = assemble_full_configuration(
        full_chain,
        (left_chain, left_default),
        (right_chain, right_default),
    )
    left_default_pose, right_default_pose = compute_dual_arm_end_effector_poses(full_chain, default_config)
    """

    joint_names = full_chain.get_joint_parameter_names(exclude_fixed=True)
    full_config = torch.zeros(len(joint_names), dtype=full_chain.dtype, device=full_chain.device)
    name_to_index: Dict[str, int] = {name: idx for idx, name in enumerate(joint_names)}

    def _insert(chain: pk.SerialChain, config: torch.Tensor) -> None:
        cfg_flat = config.detach().to(device=full_chain.device, dtype=full_chain.dtype).view(-1)
        for joint_name, value in zip(chain.get_joint_parameter_names(exclude_fixed=True), cfg_flat):
            full_config[name_to_index[joint_name]] = value

    _insert(*partial_configs)
    for chain_cfg in additional_partials:
        _insert(*chain_cfg)

    return full_config

def _summarize_results(
    chain: pk.SerialChain, solution: pk.IKSolution, targets: pk.Transform3d, arm_name: str
,verbose:bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    converged_any = solution.converged_any
    success_ids = torch.nonzero(converged_any, as_tuple=False).flatten()
    total = converged_any.numel()
    if verbose:
        print(f"{arm_name} arm: {success_ids.numel()} / {total} target poses converged.")

    if success_ids.numel() == 0:
        return success_ids, None

    best_attempts = []
    best_configs = []
    for idx in success_ids.tolist():
        # Find first converged jont angle solution
        attempt_ids = torch.nonzero(solution.converged[idx], as_tuple=False)
        best_idx = attempt_ids[0].item()
        best_attempts.append(best_idx)
        best_configs.append(solution.solutions[idx, best_idx])

    best_configs_tensor = torch.stack(best_configs, dim=0)
    fk = chain.forward_kinematics(best_configs_tensor)

    target_matrix = targets.get_matrix()[success_ids]
    fk_matrix = fk.get_matrix()

    pos_err = (target_matrix[:, :3, 3] - fk_matrix[:, :3, 3]).norm(dim=1)

    
    goal_quat = rc.matrix_to_quaternion(target_matrix[:, :3, :3])
    fk_quat = rc.matrix_to_quaternion(fk_matrix[:, :3, :3])

    # Measure distance by checking how different q_goal* inv(q_goal) is, as this results in unit quat if eqal
    rot_err = rc.quaternion_to_axis_angle(rc.quaternion_multiply(goal_quat, rc.quaternion_invert(fk_quat))).norm(dim=1)
    if verbose:
        for display_idx, goal_idx in enumerate(success_ids.tolist()):
            print(
                f"  Goal {goal_idx}: first converged retry {best_attempts[display_idx]}, "
                f"pos error {pos_err[display_idx].item():.4f} m, "
                f"rot error {rot_err[display_idx].item():.4f} rad"
            )

    return success_ids, best_configs_tensor


def _sample_reachable_targets(chain: pk.SerialChain, joint_limits: torch.Tensor, num_targets: int = 3):
    """Sample random joint configurations and compute the corresponding end-effector poses."""
    low, high = joint_limits
    q = torch.rand(num_targets, chain.n_joints, device=chain.device, dtype=chain.dtype) * (high - low) + low
    goal_poses = chain.forward_kinematics(q)
    return q, goal_poses