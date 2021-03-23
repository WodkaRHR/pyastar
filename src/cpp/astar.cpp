#include <queue>
#include <limits>
#include <cmath>
#include <Python.h>
#include <numpy/arrayobject.h>
#include <iostream>
#include <ctype.h>
#include <string>
#include <stack>
#include "grid.h"
#include "heuristic.h"

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

  int heuristic_type = parse_heuristic(str_heuristic);
  if (heuristic_type == INVALID_HEURISTIC)
    return NULL;

  int* paths = new int[h * w];
  int path_length = -1;
  bool path_exists = false;

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
      path_exists = true;
      break;
    }

    nodes_to_visit.pop();
    get_neighbours(nbrs, cur.idx, w, h);

    float heuristic_cost = 0;
    for (int i = 0; i < 8; ++i) {
      if (nbrs[i] >= 0) {
        if (weights[nbrs[i]] != 0.0) {
          if (verbose) std::cout << "\tNeighbour at " << (nbrs[i] / w) << "," << (nbrs[i] % w) << " non traversable\n";
          continue; // Non-traversable neighbour
        }
          
        // The sum of the cost so far and the cost of this move
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
  if (path_exists) {
    // Find the path length, as for non admissable heuristics the node attribute may incorrect
    int idx = goal;
    path_length = 1;
    while (idx != start && idx != -1) {
      path_length++;
      idx = paths[idx];
    }
    npy_intp dims[2] = {path_length, 2};
    PyArrayObject* path = (PyArrayObject*) PyArray_SimpleNew(2, dims, NPY_INT32);
    npy_int32 *iptr, *jptr;
    idx = goal;
    if (verbose)
      std::cout << "Reconstruction from start " << (start / w) << "," << (start % w) << " to goal " << (goal / w) << "," << (goal % w) << "\n";
    for (npy_intp i = dims[0] - 1; i >= 0; --i) {
        iptr = (npy_int32*) (path->data + i * path->strides[0]);
        jptr = (npy_int32*) (path->data + i * path->strides[0] + path->strides[1]);

        *iptr = idx / w;
        *jptr = idx % w;

        if (verbose)
          std::cout << "Reconstructing from " << (idx / w) << "," << (idx % w) << "\n";
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
