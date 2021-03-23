import ctypes
import numpy as np
import pyastar.dijkstra
from typing import Optional, Tuple


# Define array types
ndmat_f_type = np.ctypeslib.ndpointer(
    dtype=np.float32, ndim=1, flags="C_CONTIGUOUS")
ndmat_i_type = np.ctypeslib.ndpointer(
    dtype=np.int32, ndim=1, flags="C_CONTIGUOUS")
ndmat_f2_type = np.ctypeslib.ndpointer(
    dtype=np.float32, ndim=2, flags="C_CONTIGUOUS")

# Define input/output types
pyastar.dijkstra.restype = ctypes.py_object  # HxW shortest path costs
pyastar.dijkstra.argtypes = [
    ndmat_i_type,   # weights
    ctypes.c_int,   # height
    ctypes.c_int,   # width
    ctypes.c_int,   # start index in flattened grid
    ctypes.c_int,   # verbose
]

def dijkstra_costs(
        maze: np.ndarray,
        start: Tuple[int, int],
        verbose : bool = False) -> Optional[np.ndarray]:
    if maze.dtype != np.int32:
        print(f'Grid is required to have np.int32 data type, but has {maze.dtype}. Casting to np.int32')
        maze = maze.astype(np.int32)
    # Ensure start is within bounds.
    if (start[0] < 0 or start[0] >= maze.shape[0] or
            start[1] < 0 or start[1] >= maze.shape[1]):
        raise ValueError(f"Start of {start} lies outside grid.")

    height, width = maze.shape
    start_idx = np.ravel_multi_index(start, (height, width))

    costs = pyastar.dijkstra.dijkstra(
        maze.flatten(), height, width, start_idx, verbose
    )
    return costs
