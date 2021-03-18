import ctypes
import numpy as np
import pyastar.astar
from typing import Optional, Tuple


# Define array types
ndmat_f_type = np.ctypeslib.ndpointer(
    dtype=np.float32, ndim=1, flags="C_CONTIGUOUS")
ndmat_i_type = np.ctypeslib.ndpointer(
    dtype=np.int32, ndim=1, flags="C_CONTIGUOUS")
ndmat_i2_type = np.ctypeslib.ndpointer(
    dtype=np.int32, ndim=2, flags="C_CONTIGUOUS")

# Define input/output types
pyastar.astar.restype = ndmat_i2_type  # Nx2 (i, j) coordinates or None
pyastar.astar.argtypes = [
    ndmat_i_type,   # weights
    ctypes.c_int,   # height
    ctypes.c_int,   # width
    ctypes.c_int,   # start index in flattened grid
    ctypes.c_int,   # goal index in flattened grid
    ctypes.c_char_p,  # heuristic
    ndmat_f_type,   # heuristic heatmap
]

valid_heuristics = ('l1', 'l2', 'octile', 'custom')


def best_first_search_path(
        maze: np.ndarray,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        heuristic: str = "l2",
        heuristic_heatmap : Optional[np.ndarray] = None) -> Optional[np.ndarray]:
    if maze.dtype != np.int32:
        print(f'Grid is required to have np.int32 data type, but has {maze.dtype}. Casting to np.int32')
        maze = maze.astype(np.int32)
    if heuristic_heatmap is not None and heuristic_heatmap.dtype != np.float32:
        print(f'Heuristic is required to have np.float32 data type, but has {heuristic_heatmap.dtype}. Casting to np.float32')
        heuristic_heatmap = heuristic_heatmap.astype(np.float32)
    heuristic = heuristic.lower()
    if heuristic not in valid_heuristics:
        print(f'Heuristic must be one of {valid_heuristics} but is {heuristic}.')
        return None
    # Ensure start is within bounds.
    if (start[0] < 0 or start[0] >= maze.shape[0] or
            start[1] < 0 or start[1] >= maze.shape[1]):
        raise ValueError(f"Start of {start} lies outside grid.")
    # Ensure goal is within bounds.
    if (goal[0] < 0 or goal[0] >= maze.shape[0] or
            goal[1] < 0 or goal[1] >= maze.shape[1]):
        raise ValueError(f"Goal of {goal} lies outside grid.")

    height, width = maze.shape
    start_idx = np.ravel_multi_index(start, (height, width))
    goal_idx = np.ravel_multi_index(goal, (height, width))

    if heuristic_heatmap is not None:
        heuristic_heatmap = heuristic_heatmap.flatten()

    path = pyastar.astar.best_first_search(
        maze.flatten(), height, width, start_idx, goal_idx, heuristic, heuristic_heatmap,
    )
    return path
