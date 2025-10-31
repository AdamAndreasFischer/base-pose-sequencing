import torch
import numpy as np
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import multiprocessing
import threading
from heapq import heappush, heappop
import matplotlib.pyplot as plt

NUM_THREADS = max(1, int(multiprocessing.cpu_count()/2))

def parallel_Astar(costmaps, start, goal, clearance): 
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
                                   clearance ,ids, stop_event) for ids in range(costmaps.shape[0])]
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

def is_valid_point(point, grid, clearance):
    """
    Check if a point is valid (within bounds, not an obstacle, and maintains clearance).
    """
    x, y = point
    rows, cols = grid.shape

    # Check bounds
    if x < 0 or x >= rows or y < 0 or y >= cols:
        return False

    # Check clearance
    x_min = max(0, x - clearance)
    x_max = min(rows, x + clearance + 1)
    y_min = max(0, y - clearance)
    y_max = min(cols, y + clearance + 1)
    

    if np.any(grid[x_min:x_max, y_min:y_max] == 1):  # 1 represents an obstacle
        return False

    return True


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

    while open_set:
        if stop_event is not None and stop_event.is_set():
            return None
        _, current = heappop(open_set)

        # Check if goal is reached
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1], id  # Return reversed path

        for neighbor in get_neighbors(current, grid, clearance):
            tentative_g_score = g_score[current] + heuristic(current, neighbor)

            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                heappush(open_set, (f_score[neighbor], neighbor))

    return None, id  # No path found

def visualize_path(grid, path, start, goal):
    """
    Visualize the grid, obstacles, and the planned path.
    """
    plt.figure(figsize=(10, 10))
    plt.imshow(grid, cmap="gray", origin="upper")

    # Plot the path
    if path:
        path = np.array(path)
        plt.plot(path[:, 0], path[:, 1], color="red", linewidth=2, label="Path")  # Note: (y, x) order

    # Mark start and goal points
    plt.scatter(start[0], start[1], color="green", s=100, label="Start")  # (y, x) order
    plt.scatter(goal[0], goal[1], color="blue", s=100, label="Goal")  # (y, x) order

    plt.legend()
    plt.title("A* Path Planning")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    plt.show()