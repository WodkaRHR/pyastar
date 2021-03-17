import numpy as np
import pyastar


# The start and goal coordinates are in matrix coordinates (i, j).
start = (0, 0)
goal = (5, 5)

# The minimum cost must be 1 for the heuristic to be valid.
maze = np.array([   [0, 1, 1, 1, 1, 1],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 1, 0, 0],
                    [0, 0, 1, 1, 0, 0],
                    [1, 0, 1, 0, 0, 1],
                    [1, 0, 1, 1, 0, 0]], dtype=np.int32)

heuristic_map = np.random.randn(*maze.shape)

print("Heuristic:")
print(heuristic_map)
path = pyastar.astar_path(maze, start, goal, 'custom')

# The path is returned as a numpy array of (i, j) coordinates.
print(f"Shortest path from {start} to {goal} found:")
print(path)
