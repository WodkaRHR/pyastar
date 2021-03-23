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

  int heuristic_type = parse_heuristic(str_heuristic);
  if (heuristic_type == INVALID_HEURISTIC)
    return NULL;

  if (verbose) 
    std::cout << "Heuristic is " << heuristic_type << "\n";

  int* paths = new int[h * w];
  int* nbrs = new int[8];
  int *nbr_idxs = new int[8];
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
    for (int i = 0; i < 8; i++) nbr_idxs[i] = i;
    for (int iter = 0; iter < 8; iter++) {
      // Find the largest element in nbrs[iter:]
      int largest_nbr_idx = -1;
      float largest = -INF;
      for (int i = iter; i < 8; i++) {
        int next = nbrs[nbr_idxs[i]];
        if (next >= 0 && weights[next] == 0.0 && !visited[next]) {
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
          float cost = heuristic_cost + nbrs_costs[nbr_idxs[i]];
          if (cost > largest) {
            largest = cost;
            largest_nbr_idx = i;
          }
        }
      }
      if (largest_nbr_idx == -1) {
        break; // no more neighbours to exlore
      }
      int next = nbrs[nbr_idxs[largest_nbr_idx]];
      lifo.push(next);
      paths[next] = current;
      nbr_idxs[largest_nbr_idx] = nbr_idxs[iter]; // Remove the neighbour idx from the list
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
  delete[] nbr_idxs;

  return return_val;
}

static PyMethodDef bfs_methods[] = {
    {"best_first_search", (PyCFunction)best_first_search, METH_VARARGS, "best_first_search"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef bfs_module = {
    PyModuleDef_HEAD_INIT,"best_first_search", NULL, -1, bfs_methods
};

PyMODINIT_FUNC PyInit_best_first_search(void) {
  import_array();
  return PyModule_Create(&bfs_module);
}
