# Base pose sequencing from visual inputs
This is the repo of my master thesis "Base pose sequencing from visual inputs"


## Enable orthographic projection in isaac lab
In order to enable Orthographic camera projections in isaac lab 2.0.0, these modifications needs to be done in the isaac lab source code:

in `isaaclab/sim/spawners/sensors/__init__.py`:'
```
Replace
from .sensors_cfg import FisheyeCameraCfg, PinholeCameraCfg
With
from .sensors_cfg import FisheyeCameraCfg, PinholeCameraCfg, OrthographicCameraCfg
```

In `isaaclab/sim/spawners/sensors/sensors_cfg.py`:

Add a configclass:
```
@configclass
class OrthographicCameraCfg(FisheyeCameraCfg):
    func: Callable = sensors.spawn_camera
    projection_type: Literal[
        "fisheyePolynomial",
        "fisheyeSpherical",
        "fisheyeKannalaBrandtK3",
        "fisheyeRadTanThinPrism", 
        "omniDirectionalStereo", 
        ] = "fisheyePolynomial"

    projection: str = "orthographic"
```

In `isaaclab/sim/spawners/sensors/sensors.py` copy the Custom fisheye camera attributes and add ` "projection":("projection", Sdf.ValueTypeNames.Token) ` to it. Add the attributes as an option in spawn_camera.

In `isaaclab/sim/spawners/sensors/camera/camera_cfg.py` import the class and add it to the spawn line. 
