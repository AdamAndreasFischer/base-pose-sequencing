from functools import lru_cache


@lru_cache(maxsize=1)
def parse_prim_paths(path):
    """Parses path of prims in rigid object collection"""
    parts = path.split('/')
    env_id = int(parts[3].split('_')[1])
    obj_id=int(parts[4].split('_')[1])
    return(env_id, obj_id)