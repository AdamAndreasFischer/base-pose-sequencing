# Example using NVIDIA Isaac Sim's tools (requires Isaac Sim Python environment)
#import omni.isaac.urdf as urdf_extension

from isaacsim.asset.importer.urdf import _urdf

# Setup importer
urdf_interface = urdf_extension.acquire_urdf_interface()

# Path to your URDF file and destination
urdf_path = "/home/adamfi/codes/base-pose-sequencing/conf/motion/robot.urdf"
output_path = "/home/adamfi"

# Import settings
import_config = urdf_extension.ImportConfig()
import_config.fix_base = False  # Set to True for fixed base robots
import_config.make_default_prim = True



# Perform the conversion
status, usd_path = urdf_interface.import_urdf(urdf_path, output_path, import_config)

