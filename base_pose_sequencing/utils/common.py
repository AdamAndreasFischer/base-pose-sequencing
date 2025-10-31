from functools import lru_cache
from isaaclab.managers import SceneEntityCfg
import torch
import matplotlib.pyplot as plt

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


def pose_to_pixel(env, observation_space, poses):
    """
    Top left of scene is 0,0 in pixels, bottom right is 160,160
    """
    x,y = poses[:,0], poses[:,1]

    max_coord = 2.5

    W,H = observation_space.shape[2], observation_space.shape[3]

    x_pix = ((x+max_coord)/(2*max_coord)*W).to(dtype=torch.int32)
    y_pix = ((max_coord-y)/(2*max_coord)*H).to(dtype=torch.int32)

    poses = torch.cat([x_pix.unsqueeze(-1),y_pix.unsqueeze(-1)], dim=1).to(device=env.device)

    return poses

def visualize_scene(camera, env_id):
    rgb = camera.data.output["rgb"][env_id].cpu().numpy()
    
    plt.imshow(rgb)

    plt.show()
