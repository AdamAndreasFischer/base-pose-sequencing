# Base pose sequencing from visual inputs
This is the repo of my master thesis "Base pose sequencing from visual inputs"


## Enable orthographic projection in isaac lab
In order to enable Orthographic camera projections in isaac lab 2.0.0, these modifications needs to be done in the isaac lab source code:

in `isaaclab/isaaclab/sim/spawners/sensors/__init__.py`:'
```
Replace
from .sensors_cfg import FisheyeCameraCfg, PinholeCameraCfg
With
from .sensors_cfg import FisheyeCameraCfg, PinholeCameraCfg, OrthographicCameraCfg
```

In `isaaclab/isaaclab/sim/spawners/sensors/sensors_cfg.py`:

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

In `isaaclab/isaaclab/sim/spawners/sensors/sensors.py` add

```
CUSTOM_ORTHOGRAPHIC_CAMERA_ATTRIBUTES = {
    "projection_type": ("cameraProjectionType", Sdf.ValueTypeNames.Token),
    "fisheye_nominal_width": ("fthetaWidth", Sdf.ValueTypeNames.Float),
    "fisheye_nominal_height": ("fthetaHeight", Sdf.ValueTypeNames.Float),
    "fisheye_optical_centre_x": ("fthetaCx", Sdf.ValueTypeNames.Float),
    "fisheye_optical_centre_y": ("fthetaCy", Sdf.ValueTypeNames.Float),
    "fisheye_max_fov": ("fthetaMaxFov", Sdf.ValueTypeNames.Float),
    "fisheye_polynomial_a": ("fthetaPolyA", Sdf.ValueTypeNames.Float),
    "fisheye_polynomial_b": ("fthetaPolyB", Sdf.ValueTypeNames.Float),
    "fisheye_polynomial_c": ("fthetaPolyC", Sdf.ValueTypeNames.Float),
    "fisheye_polynomial_d": ("fthetaPolyD", Sdf.ValueTypeNames.Float),
    "fisheye_polynomial_e": ("fthetaPolyE", Sdf.ValueTypeNames.Float),
    "fisheye_polynomial_f": ("fthetaPolyF", Sdf.ValueTypeNames.Float),
    "projection": ("projection", Sdf.ValueTypeNames.Token)
}
```
In arguments to spawn_camera make following changes
```
cfg: sensors_cfg.PinholeCameraCfg | sensors_cfg.FisheyeCameraCfg,
***to***
cfg: sensors_cfg.PinholeCameraCfg | sensors_cfg.FisheyeCameraCfg| sensors_cfg.OrthographicCameraCfg,
```
In function change
```
if cfg.projection_type == "pinhole":
        attribute_types = CUSTOM_PINHOLE_CAMERA_ATTRIBUTES
else:
    attribute_types = CUSTOM_FISHEYE_CAMERA_ATTRIBUTES
***to***
if cfg.projection_type == "pinhole":
    attribute_types = CUSTOM_PINHOLE_CAMERA_ATTRIBUTES
elif cfg.projection == "orthographic":
    attribute_types = CUSTOM_ORTHOGRAPHIC_CAMERA_ATTRIBUTES
else:
    attribute_types = CUSTOM_FISHEYE_CAMERA_ATTRIBUTES
```


In `isaaclab/isaaclab/sensors/camera/camera_cfg.py` import the class and add it to the spawn line in the class CameraCfg. 
