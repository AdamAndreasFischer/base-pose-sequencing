import torch
import numpy as np
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import multiprocessing
import threading
from heapq import heappush, heappop
import matplotlib.pyplot as plt
from numba import jit
NUM_THREADS = max(1, int(multiprocessing.cpu_count()/2))

def parallel_Astar(costmaps, start, goal, clearance, env_ids): 
    """
    Try to find way to optimize, it is still too slow
    """
    manager = multiprocessing.Manager()
    costmaps = costmaps.cpu().numpy()
    start = start.cpu().tolist()
    goal = goal.cpu().tolist()
    paths = {}
    stop_event = manager.Event()

    with ProcessPoolExecutor(max_workers=NUM_THREADS) as executor:
        futures = [executor.submit(a_star, costmaps[ids], 
                                   (start[ids][0], start[ids][1]), 
                                   (goal[ids][0],goal[ids][1]), 
                                   clearance ,ids, stop_event) for ids in range(env_ids.shape[0])]
        try:
            for future in futures:
                result = future.result()
                if result is not None:
                    path, env_id = result
                    paths[env_id] = path

        except KeyboardInterrupt:
            print("ctrl+c detected")
            stop_event.set()
            for future in futures:
                future.cancel()
            executor.shutdown(wait=False, cancel_futures=True)
            raise
        except Exception as e:
            stop_event.set()
            for future in futures:
                future.cancel()
            executor.shutdown(wait=False, cancel_futures=True)
            raise
    
    
    
    return paths
@jit
def is_valid_point(point, grid, clearance):
    """
    Check if a point is valid (within bounds, not an obstacle, and maintains clearance).
    Uses preinflated maps, reduces to one operation
    """
    x, y = point
    rows, cols = grid.shape

    # Check bounds
    if x < 0 or x >= rows or y < 0 or y >= cols:
        return False

    # Check clearance
    #x_min = max(0, x - clearance)
    #x_max = min(rows, x + clearance + 1)
    #y_min = max(0, y - clearance)
    #_max = min(cols, y + clearance + 1)
    

    #if np.any(grid[x_min:x_max, y_min:y_max] == 1):  # 1 represents an obstacle
    #    return False

    return grid[x,y] == 0

@jit
def get_neighbors(point, grid, clearance):
    """
    Get valid neighbors of a point, considering clearance.
    """
    x, y = point
    neighbors = [
        (x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1),  # Cardinal directions
        (x - 1, y - 1), (x - 1, y + 1), (x + 1, y - 1), (x + 1, y + 1)  # Diagonal directions
    ]
    return [n for n in neighbors if is_valid_point(n, grid, clearance)]

@jit
def heuristic(a, b):
    """
    Heuristic function for A* (Euclidean distance).
    """
    return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def a_star(grid, start, goal, clearance, id, stop_event=None):
    """
    A* path planning algorithm with clearance.
    """

    rows, cols = grid.shape
    open_set = []
    heappush(open_set, (0, start))  # (priority, point)
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}
    closed = set()

    while open_set:
        if stop_event is not None and stop_event.is_set():
            return None
        _, current = heappop(open_set)

        # Check if goal is reached
        if current in closed:
            continue
        closed.add(current)
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1], id  # Return reversed path

        for neighbor in get_neighbors(current, grid, clearance):
            if neighbor in closed:
                continue
            tentative_g_score = g_score[current] + heuristic(current, neighbor)

            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                heappush(open_set, (f_score[neighbor], neighbor))

    return None, id  # No path found


#@jit(nopython=False)
def a_star_numpy(grid, start, goal, clearance, id, stop_event=None): # Slower then previous version
    """
    A* path planning algorithm with clearance.
    """
    inf = 10e6
    rows, cols = grid.shape
    open_set = []
    heappush(open_set, (0, start))  # (priority, point)
    #came_from = {}
    parent_x = np.full((rows, cols), -1, dtype=np.int32)
    parent_y = np.full((rows, cols), -1, dtype=np.int32)
    
    #g_score = {start: 0}
    #f_score = {start: heuristic(start, goal)}
    g_score = np.full(grid.shape, inf, dtype=np.float32)
    f_score = np.full_like(grid, inf)
    g_score[start[0], start[1]] = 0
    f_score[start[0],start[1]] = heuristic(start, goal)
    closed = np.zeros(grid.shape, dtype=bool)
    while open_set:
        if stop_event is not None and stop_event.is_set():
            return None
        _,current= heappop(open_set)
        

        if closed[current[0], current[1]]:  # <-- ADD THIS CHECK
            continue
        closed[current[0], current[1]] = True  # <-- MARK AS CLOSED
        # Check if goal is reached
        if current == goal:
            path = []
            while not (current[0]==start[0] and current[1]==start[1]) :
                path.append(current)
                px = parent_x[current[0], current[1]]
                py = parent_y[current[0], current[1]]
                current = (px, py)
            path.append(start)
            return path[::-1], id  # Return reversed path

        for neighbor in get_neighbors(current, grid, clearance):
            if closed[neighbor[0], neighbor[1]]:  # <-- SKIP CLOSED NEIGHBORS
                continue
            tentative_g_score = g_score[current[0],current[1]] + heuristic(current, neighbor)

            if tentative_g_score < g_score[neighbor[0], neighbor[1]]: #neighbor not in g_score or 
                #came_from[neighbor] = current
                parent_x[neighbor[0], neighbor[1]] = current[0]
                parent_y[neighbor[0], neighbor[1]] = current[1]

                g_score[neighbor[0], neighbor[1]] = tentative_g_score
                f_score[neighbor[0], neighbor[1]] = tentative_g_score + heuristic(neighbor, goal)
                heappush(open_set, (f_score[neighbor[0], neighbor[1]], neighbor))

    return None, id  # No path found

def visualize_path(grid, path, start, goal, env, env_id=0):
    """Visualize the costmap path alongside the corresponding RGB frame."""
    camera = env.scene["camera"]
    rgb_frame = camera.data.output["rgb"][int(env_id)]
    if torch.is_tensor(rgb_frame):
        rgb_frame = rgb_frame.cpu().numpy()

    if torch.is_tensor(grid):
        grid = grid.cpu().numpy()

    fig, (ax_path, ax_rgb) = plt.subplots(1, 2, figsize=(14, 7))

    ax_path.imshow(grid, cmap="gray", origin="upper")

    if path:
        path_arr = np.array(path)
        ax_path.plot(path_arr[:, 0], path_arr[:, 1], color="red", linewidth=2, label="Path")

    ax_path.scatter(start[0], start[1], color="green", s=100, label="Start")
    ax_path.scatter(goal[0], goal[1], color="blue", s=100, label="Goal")
    ax_path.set_title("A* Path Planning")
    ax_path.set_xlabel("X")
    ax_path.set_ylabel("Y")
    ax_path.grid(True)
    ax_path.legend()

    ax_rgb.imshow(rgb_frame)
    ax_rgb.set_title("RGB Frame")
    ax_rgb.axis("off")

    plt.tight_layout()
    plt.show()