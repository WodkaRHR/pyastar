#ifndef H_GRID
#define H_GRID

#include <string>
#include <cmath>
#include <limits>

const float INF = std::numeric_limits<float>::infinity();

// represents a single pixel
class Node {
  public:
    int idx; // index in the flattened grid
    float cost; // cost of traversing this pixel
    int path_length; // the length of the path to reach this node

    Node(int i, float c, int path_length) : idx(i), cost(c), path_length(path_length) {}
};

// the top of the priority queue is the greatest element by default,
// but we want the largest, so flip the sign
bool operator<(const Node &n1, const Node &n2) {
  return n1.cost > n2.cost;
}

// See for various grid heuristics:
// http://theory.stanford.edu/~amitp/GameProgramming/Heuristics.html#S7
// L_\inf norm (diagonal distance)
inline float linf_norm(int i0, int j0, int i1, int j1) {
  return std::max(std::abs(i0 - i1), std::abs(j0 - j1));
}

// L_1 norm (manhattan distance)
inline float l1_norm(int i0, int j0, int i1, int j1) {
  return std::abs(i0 - i1) + std::abs(j0 - j1);
}

// L_2 norm (euclidean distance)
inline float l2_norm(int i0, int j0, int i1, int j1) {
  int dx = i0 - i1;
  int dy = j0 - j1;
  return sqrt(dx * dx + dy * dy);
}

// octile distance
inline float octile_cost(int i0, int j0, int i1, int j1) {
  int dx = std::abs(i0 - i1);
  int dy = std::abs(j0 - j1);
  if (dx >= dy) {
    return dx + (M_SQRT2 - 1) * dy;
  } else {
    return dy + (M_SQRT2 - 1) * dx;
  }
}

inline void get_neighbours(int *nbrs, int current, int w, int h) {
  int row = current / w;
  int col = current % w;
  // check bounds and find up to eight neighbors: top to bottom, left to right
  nbrs[0] = (row > 0 && col > 0)                     ? current - w - 1   : -1;
  nbrs[1] = (row > 0)                                ? current - w       : -1;
  nbrs[2] = (row > 0 && col + 1 < w)                 ? current - w + 1   : -1;
  nbrs[3] = (col > 0)                                ? current - 1       : -1;
  nbrs[4] = (col + 1 < w)                            ? current + 1       : -1;
  nbrs[5] = (row + 1 < h && col > 0)                 ? current + w - 1   : -1;
  nbrs[6] = (row + 1 < h)                            ? current + w       : -1;
  nbrs[7] = (row + 1 < h && col + 1 < w )            ? current + w + 1   : -1;
}

// Traversal costs for moving to each adjacent neighbour
const float nbrs_costs[8] = {
  M_SQRT2, 1, M_SQRT2, 1, 1, M_SQRT2, 1, M_SQRT2,
};

#endif