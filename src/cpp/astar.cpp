#include <queue>
#include <limits>
#include <cmath>
#include <Python.h>
#include <numpy/arrayobject.h>
#include <iostream>
#include <ctype.h>
#include <string>

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
// but we want the smallest, so flip the sign
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

enum {
  L2_HEURISTIC,
  L1_HEURISTIC,
  OCTILE_HEURISTIC,
  CUSTOM_HEURISTIC
};

const char str_l2_heuristic[] = "l2";
const char str_l1_heuristic[] = "l1";
const char str_octile_heuristic[] = "octile";
const char str_custom_heuristic[] = "custom";

const float nbrs_costs[8] = {
  M_SQRT2, 1, M_SQRT2, 1, 1, M_SQRT2, 1, M_SQRT2,
};


// weights:        flattened h x w grid of costs
// h, w:           height and width of grid
// start, goal:    index of start/goal in flattened grid
// diag_ok:        if true, allows diagonal moves (8-conn.)
// paths (output): for each node, stores previous node in path
static PyObject *astar(PyObject *self, PyObject *args) {
  const PyArrayObject* weights_object;
  const PyArrayObject* heuristic_object;
  int h;
  int w;
  int start;
  int goal;
  const char *str_heuristic;

  if (!PyArg_ParseTuple(
        args, "OiiiisO", // i = int, O = object
        &weights_object,
        &h, &w,
        &start, &goal,
        &str_heuristic,
        &heuristic_object))
    return NULL;

  int* weights = (int*) weights_object->data;
  float* heuristic_map = NULL;
  if (heuristic_object)
    heuristic_map = (float*) heuristic_object->data;

  int heuristic_type;
  if (strcmp(str_heuristic, str_l2_heuristic) == 0) {
    heuristic_type = L2_HEURISTIC;
  } else if (strcmp(str_heuristic, str_l1_heuristic) == 0) {
    heuristic_type = L1_HEURISTIC;
  } else if (strcmp(str_heuristic, str_octile_heuristic) == 0) {
    heuristic_type = OCTILE_HEURISTIC;
  } else if (strcmp(str_heuristic, str_custom_heuristic) == 0) {
    heuristic_type = CUSTOM_HEURISTIC;
  } else {
    // std::cout << "No valid heuristic specified";
    return NULL;
  }

  // std::cout << "Heuristic is " << heuristic_type << "\n";

  int* paths = new int[h * w];
  int path_length = -1;

  Node start_node(start, 0., 1);

  float* costs = new float[h * w];
  for (int i = 0; i < h * w; ++i)
    costs[i] = INF;
  costs[start] = 0.;

  std::priority_queue<Node> nodes_to_visit;
  nodes_to_visit.push(start_node);

  int* nbrs = new int[8];

  while (!nodes_to_visit.empty()) {
    // .top() doesn't actually remove the node
    Node cur = nodes_to_visit.top();

    if (cur.idx == goal) {
      path_length = cur.path_length;
      break;
    }

    nodes_to_visit.pop();

    int row = cur.idx / w;
    int col = cur.idx % w;
    // std::cout << "Dequeueing " << row << "," << col << "\n";

    // check bounds and find up to eight neighbors: top to bottom, left to right
    nbrs[0] = (row > 0 && col > 0)                     ? cur.idx - w - 1   : -1;
    nbrs[1] = (row > 0)                                ? cur.idx - w       : -1;
    nbrs[2] = (row > 0 && col + 1 < w)                 ? cur.idx - w + 1   : -1;
    nbrs[3] = (col > 0)                                ? cur.idx - 1       : -1;
    nbrs[4] = (col + 1 < w)                            ? cur.idx + 1       : -1;
    nbrs[5] = (row + 1 < h && col > 0)                 ? cur.idx + w - 1   : -1;
    nbrs[6] = (row + 1 < h)                            ? cur.idx + w       : -1;
    nbrs[7] = (row + 1 < h && col + 1 < w )            ? cur.idx + w + 1   : -1;

    float heuristic_cost = 0;
    for (int i = 0; i < 8; ++i) {
      if (nbrs[i] >= 0) {
        // int nx = nbrs[i] / w;
        // int ny = nbrs[i] % w;

        if (weights[nbrs[i]]) {
          // std::cout << "\tNeighbour at " << nx << "," << ny << " non traversable\n";
          continue; // Non-traversable neighbour
        }
          
        // the sum of the cost so far and the cost of this move
        float new_cost = costs[cur.idx] + nbrs_costs[i];
        if (new_cost < costs[nbrs[i]]) {
          switch (heuristic_type) {
            case L2_HEURISTIC:
              heuristic_cost = l2_norm(nbrs[i] / w, nbrs[i] % w,
                                     goal    / w, goal    % w);
              break;
            case L1_HEURISTIC: 
              heuristic_cost = l1_norm(nbrs[i] / w, nbrs[i] % w,
                                     goal    / w, goal    % w);
              break;
            case OCTILE_HEURISTIC:
              heuristic_cost = octile_cost(nbrs[i] / w, nbrs[i] % w,
                                     goal    / w, goal    % w);
              break;
            case CUSTOM_HEURISTIC:
              heuristic_cost = heuristic_map[nbrs[i]];
              break;
          }
          // paths with lower expected cost are explored first
          float priority = new_cost + heuristic_cost;
          nodes_to_visit.push(Node(nbrs[i], priority, cur.path_length + 1));
          // std::cout << "\tNeighbour at " << nx << "," << ny << " enqueued with cost " << new_cost << " and heuristic " << heuristic_cost << "\n";

          costs[nbrs[i]] = new_cost;
          paths[nbrs[i]] = cur.idx;
        }
      }
    }
  }

  PyObject *return_val;
  if (path_length >= 0) {
    npy_intp dims[2] = {path_length, 2};
    PyArrayObject* path = (PyArrayObject*) PyArray_SimpleNew(2, dims, NPY_INT32);
    npy_int32 *iptr, *jptr;
    int idx = goal;
    for (npy_intp i = dims[0] - 1; i >= 0; --i) {
        iptr = (npy_int32*) (path->data + i * path->strides[0]);
        jptr = (npy_int32*) (path->data + i * path->strides[0] + path->strides[1]);

        *iptr = idx / w;
        *jptr = idx % w;

        idx = paths[idx];
    }

    return_val = PyArray_Return(path);
  }
  else {
    return_val = Py_BuildValue(""); // no soln --> return None
  }

  delete[] costs;
  delete[] nbrs;
  delete[] paths;

  return return_val;
}

static PyMethodDef astar_methods[] = {
    {"astar", (PyCFunction)astar, METH_VARARGS, "astar"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef astar_module = {
    PyModuleDef_HEAD_INIT,"astar", NULL, -1, astar_methods
};

PyMODINIT_FUNC PyInit_astar(void) {
  import_array();
  return PyModule_Create(&astar_module);
}
