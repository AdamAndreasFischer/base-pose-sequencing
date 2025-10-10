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
class OrthographicCameraCfg(PinholeCameraCfg):
    """Configuration parameters for a USD camera prim with `orthographic camera`_ settings.

    For more information on the parameters, please refer to the
    `camera documentation <https://docs.omniverse.nvidia.com/materials-and-rendering/latest/cameras.html#fisheye-properties>`__.

    .. note::
        The default values are taken from the `Replicator camera <https://docs.omniverse.nvidia.com/py/replicator/1.9.8/source/omni.replicator.core/docs/API.html#omni.replicator.core.create.camera>`__
        function.

    .. _fish-eye camera: https://en.wikipedia.org/wiki/Fisheye_lens
    """
    func: Callable = sensors.spawn_camera

    projection: str = "orthographic"
```

In `isaaclab/sim/spawners/sensors/sensors.py` copy the Custom fisheye camera attributes and add ` "projection":("projection", Sdf.ValueTypeNames.Token) ` to it. Add the attributes as an option in spawn_camera.

In `isaaclab/sim/spawners/sensors/camera/camera_cfg.py` import the class and add it to the spawn line in the class CameraCfg. 
