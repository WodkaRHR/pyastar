#include <queue>
#include <limits>
#include <cmath>
#include <Python.h>
#include <numpy/arrayobject.h>
#include <iostream>
#include <ctype.h>
#include <string>
#include <stack>

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
  int verbose;

  if (!PyArg_ParseTuple(
        args, "OiiiisOi", // i = int, O = object
        &weights_object,
        &h, &w,
        &start, &goal,
        &str_heuristic,
        &heuristic_object,
        &verbose))
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
    get_neighbours(nbrs, cur.idx, w, h);

    float heuristic_cost = 0;
    for (int i = 0; i < 8; ++i) {
      if (nbrs[i] >= 0) {
        if (weights[nbrs[i]]) {
          if (verbose) std::cout << "\tNeighbour at " << (nbrs[i] / w) << "," << (nbrs[i] % w) << " non traversable\n";
           
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
          if (verbose) {
            std::cout << "\tNeighbour at " << (nbrs[i] / w) << "," << (nbrs[i] % w) << " enqueued with cost " << new_cost << " and heuristic " << heuristic_cost << "\n";
          } 

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

static PyObject *best_first_search(PyObject *self, PyObject *args) {
  const PyArrayObject* weights_object;
  const PyArrayObject* heuristic_object;
  int h;
  int w;
  int start;
  int goal;
  const char *str_heuristic;
  int verbose;

  if (!PyArg_ParseTuple(
        args, "OiiiisOi", // i = int, O = object
        &weights_object,
        &h, &w,
        &start, &goal,
        &str_heuristic,
        &heuristic_object,
        &verbose))
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
    if (verbose) std::cout << "No valid heuristic specified";
    return NULL;
  }

  if (verbose) std::cout << "Heuristic is " << heuristic_type << "\n";

  int* paths = new int[h * w];
  int* nbrs = new int[8];
  std::vector<bool> visited(h * w, false);
  std::stack<int> lifo;
  lifo.push(start);

  bool path_exists = false;

  while (!lifo.empty()) {
    int current = lifo.top();
    lifo.pop();
    if (verbose) std::cout << "Popped " << (current / w) << "," << (current % w) << "\n";
    if (visited[current])
      continue;
    visited[current] = true;
    if (current == goal) {
      path_exists = true;
      break;
    }
    get_neighbours(nbrs, current, w, h);
    // Push all univisted neighbours in order of increasing heuristic, for small array of size 8 just use insertion sort
    for (int i = 0; i < 8; i++) {
      float largest = -INF;
      int largest_idx = -1;
      for (int j = i; j < 8; j++) {
        int next = nbrs[j];
        if (verbose && next != -1) std::cout << "Insertion iter " << i << ": Considering " << (next / w) << "," << (next % w) << " with weight " << weights[next] << "\n";
        if (next != -1 && !weights[next] && !visited[next]) {
          float heuristic_cost = INF;
          switch (heuristic_type) {
            case L2_HEURISTIC:
              heuristic_cost = l2_norm(next / w, next % w,
                                     goal    / w, goal    % w);
              break;
            case L1_HEURISTIC: 
              heuristic_cost = l1_norm(next / w, next % w,
                                     goal    / w, goal    % w);
              break;
            case OCTILE_HEURISTIC:
              heuristic_cost = octile_cost(next / w, next % w,
                                     goal    / w, goal    % w);
              break;
            case CUSTOM_HEURISTIC:
              heuristic_cost = heuristic_map[next];
              break;
          }
          if (verbose) std::cout << "\thas heuristic cost: " << heuristic_cost << "\n";
          if (heuristic_cost > largest) {
            largest = heuristic_map[next];
            largest_idx = j;
          }
        }
      }
      if (largest_idx == -1)
        break;
      lifo.push(nbrs[largest_idx]);
      if (verbose) std::cout << "Pushed " << (nbrs[largest_idx] / w) << "," << (nbrs[largest_idx] % w) << " with heuristic of " << heuristic_map[nbrs[largest_idx]] << "\n";
      paths[nbrs[largest_idx]] = current;
      // Remove the neighbour at largest_idx by swapping with the first element
      nbrs[largest_idx] = nbrs[i];
    }
  }

  PyObject *return_val;
  if (path_exists) {
    // Find the path length
    int idx = goal;
    int path_length = 1;
    while (idx != start && idx != -1) {
      path_length++;
      idx = paths[idx];
    }
    npy_intp dims[2] = {path_length, 2};
    PyArrayObject* path = (PyArrayObject*) PyArray_SimpleNew(2, dims, NPY_INT32);
    npy_int32 *iptr, *jptr;
    idx = goal;
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
  delete[] nbrs;
  delete[] paths;

  return return_val;
}

static PyMethodDef astar_methods[] = {
    {"astar", (PyCFunction)astar, METH_VARARGS, "astar"},
    {"best_first_search", (PyCFunction)best_first_search, METH_VARARGS, "best_first_search"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef astar_module = {
    PyModuleDef_HEAD_INIT,"astar", NULL, -1, astar_methods
};

PyMODINIT_FUNC PyInit_astar(void) {
  import_array();
  return PyModule_Create(&astar_module);
}
