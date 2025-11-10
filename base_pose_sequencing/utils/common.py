from functools import lru_cache
from isaaclab.managers import SceneEntityCfg
import torch
import matplotlib.pyplot as plt
from isaaclab.utils.math import euler_xyz_from_quat

@lru_cache(maxsize=1)
def parse_prim_paths(path):
    """Parses path of prims in rigid object collection"""
    parts = path.split('/')
    env_id = int(parts[3].split('_')[1])
    obj_id=int(parts[4].split('_')[1])
    return(env_id, obj_id)



def custom_image(env
        ,sensor_cfg:SceneEntityCfg= SceneEntityCfg("camera")):
    
    """Return an image on form [n_chanels, width, height]"""
    sensor = env.scene.sensors[sensor_cfg.name]
    
    for i in range(15):
        env.sim.render()
    sensor.reset()
    
    rgb  = sensor.data.output["rgb"]
    depth = sensor.data.output["distance_to_image_plane"]
    
    image = torch.cat([rgb, depth], dim=-1)

    image= image.transpose(3,1) 
 

    return image

def compare_pixel_to_world(env):
    table_poses = env.scene["robot"].data.root_link_state_w[:,:2]
    angle = env.scene["robot"].data.root_link_state_w[:,3:7] # w,x,y,z 
    sem_seg = env.scene["camera"].data.output["semantic_segmentation"].squeeze(-1)[0]
    table_map = torch.zeros_like(sem_seg)


    camera_info = env.scene["camera"].data.info

    idtolabels = camera_info[0]["semantic_segmentation"]["idToLabels"]

    for key, value in idtolabels.items():
        if "robot" in value:
            table_map[sem_seg==int(key)] = 1
    
    poses = torch.where(table_map==1)
    
    pose = (int(torch.mean(poses[1].to(dtype=torch.float32))), int(torch.mean(poses[0].to(dtype=torch.float32))))

    euler = euler_xyz_from_quat(angle)

    z_axis_rot = euler[-1]


    w_x,w_y = pixel_to_world(env,pose[0], pose[1], z_axis_rot)
    pix_x, pix_y = world_to_pixel(env, w_x, w_y, z_axis_rot)
  
    #return pose

def pixel_to_world(env,pix_x, pix_y, rotation_in_world):
    """
    Almost done, add in compensation for c.o.m of segmentation mask being closer to the yumi body. 
    """
    offset = 0.08 # Offest for unbalanced segmentation map
    camera = env.scene["camera"]
    intrinsic_matrices = camera.data.intrinsic_matrices
 
    w_x = (pix_x - intrinsic_matrices[:,0,2])/20.0 # Scaling
    w_y = -(pix_y - intrinsic_matrices[:,1,2])/20.0



    w_x = w_x-offset*torch.cos(rotation_in_world)
    w_y = w_y-offset*torch.sin(rotation_in_world)

    return w_x, w_y

def world_to_pixel(env, w_x, w_y, rotation, env_ids):

    intrinsic_matrices = env.scene["camera"].data.intrinsic_matrices[env_ids]

    offset = 0.08

    w_x_orig = w_x+offset*torch.cos(rotation)
    w_y_orig = w_y+offset*torch.sin(rotation)
    
    pix_x = intrinsic_matrices[:,0,2]+(w_x_orig*20.0)
    pix_y = intrinsic_matrices[:,1,2] - (w_y_orig*20.0)
    pix_x = pix_x.unsqueeze(1)
    pix_y = pix_y.unsqueeze(1)

    return torch.cat([pix_x,pix_y],dim=1).to(dtype=torch.int64) # This concat should be right. returns [[x,y], [x,y]]

# Deprecated
def pose_to_pixel(env, observation_space, poses):
    """
    Top left of scene is 0,0 in pixels, bottom right is 160,160
    """
    x,y = poses[:,0], poses[:,1]

    max_coord = 2.875

    W,H = observation_space.shape[2], observation_space.shape[3]

    x_pix = ((x+max_coord)/(2*max_coord)*W).to(dtype=torch.int32)
    y_pix = ((max_coord-y)/(2*max_coord)*H).to(dtype=torch.int32)

    poses = torch.cat([x_pix.unsqueeze(-1),y_pix.unsqueeze(-1)], dim=1).to(device=env.device)

    return poses

def visualize_scene(camera, env_id):
    rgb = camera.data.output["rgb"][env_id].cpu().numpy()
    
    plt.imshow(rgb)

    plt.show()


from typing import List, Dict, Tuple 
from heapq import heappush, heappop
import numba as nb
from numba.experimental import jitclass

# priority, counter, item, removed
entry_def = (0.0, 0, (0,0), nb.typed.List([False]))
entry_type = nb.typeof(entry_def)

@jitclass
class PriorityQueue:
    # The following helps numba infer type of variable
    pq: List[entry_type]
    entry_finder: Dict[Tuple[int, int], entry_type]
    counter: int
    entry: entry_type

    def __init__(self):
        # Must declare types here see https://numba.pydata.org/numba-doc/dev/reference/pysupported.html
        self.pq = nb.typed.List.empty_list((0.0, 0, (0,0), nb.typed.List([False])))
        self.entry_finder = nb.typed.Dict.empty( (0, 0), (0.0, 0, (0,0), nb.typed.List([False])))
        self.counter = 0

    def put(self, item: Tuple[int, int], priority: float = 0.0):
        """Add a new item or update the priority of an existing item"""
        if item in self.entry_finder:
            # Mark duplicate item for deletion
            self.remove_item(item)
    
        self.counter += 1
        entry = (priority, self.counter, item, nb.typed.List([False]))
        self.entry_finder[item] = entry
        heappush(self.pq, entry)

    def remove_item(self, item: Tuple[int, int]):
        """Mark an existing item as REMOVED via True.  Raise KeyError if not found."""
        self.entry = self.entry_finder.pop(item)
        self.entry[3][0] = True
    
    def pop(self):
        """Remove and return the lowest priority item. Raise KeyError if empty."""
        while self.pq:
            priority, count, item, removed = heappop(self.pq)
            if not removed[0]:
                del self.entry_finder[item]
                return priority, item
        raise KeyError("pop from an empty priority queue")