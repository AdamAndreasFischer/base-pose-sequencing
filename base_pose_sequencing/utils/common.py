from functools import lru_cache
from isaaclab.managers import SceneEntityCfg
import torch

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

    rgb  = sensor.data.output["rgb"]
    depth = sensor.data.output["distance_to_image_plane"]
    
    image = torch.cat([rgb, depth], dim=-1)

    image= image.transpose(3,1) # Transpose in order to 
 

    return image


def robot_pose_to_pixel(env, robot_poses):
    x,y = robot_poses[0], robot_poses[1]

    W,H = env.observation_space.shape[1], env.observation_space[2]

    