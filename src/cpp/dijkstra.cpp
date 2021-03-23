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

// weights:        flattened h x w grid of costs
// h, w:           height and width of grid
// start, goal:    index of start/goal in flattened grid
// diag_ok:        if true, allows diagonal moves (8-conn.)
// paths (output): for each node, stores previous node in path
static PyObject *dijkstra(PyObject *self, PyObject *args) {
  const PyArrayObject* weights_object;
  int h;
  int w;
  int start;
  int verbose;

  if (!PyArg_ParseTuple(
        args, "Oiiii", // i = int, O = object
        &weights_object,
        &h, &w,
        &start, 
        &verbose))
    return NULL;

  int* weights = (int*) weights_object->data;
  int* paths = new int[h * w];
  Node start_node(start, 0., 1);

  float* costs = new float[h * w];
  for (int i = 0; i < h * w; ++i) {
    costs[i] = INF;
    paths[i] = -1;
  }
  costs[start] = 0.;

  std::priority_queue<Node> nodes_to_visit;
  nodes_to_visit.push(start_node);
  
  int* nbrs = new int[8];

  while (!nodes_to_visit.empty()) {
    // .top() doesn't actually remove the node
    Node cur = nodes_to_visit.top();

    nodes_to_visit.pop();
    get_neighbours(nbrs, cur.idx, w, h);

    for (int i = 0; i < 8; ++i) {
      if (nbrs[i] >= 0) {
        if (weights[nbrs[i]] != 0.0) {
          if (verbose) std::cout << "\tNeighbour at " << (nbrs[i] / w) << "," << (nbrs[i] % w) << " non traversable\n";
          continue; // Non-traversable neighbour
        }
          
        // The sum of the cost so far and the cost of this move
        float new_cost = costs[cur.idx] + nbrs_costs[i];
        if (new_cost < costs[nbrs[i]]) {
          // paths with lower cost are explored first
          nodes_to_visit.push(Node(nbrs[i], new_cost, cur.path_length + 1));
          if (verbose) {
            std::cout << "\tNeighbour at " << (nbrs[i] / w) << "," << (nbrs[i] % w) << " enqueued with cost " << new_cost << "\n";
          } 
          costs[nbrs[i]] = new_cost;
          paths[nbrs[i]] = cur.idx;
        }
      }
    }
  }

  npy_intp dims[2] = {h, w};
  npy_intp dims_paths[3] = {h, w, 2};
  PyArrayObject* result_costs = (PyArrayObject*) PyArray_SimpleNew(2, dims, NPY_FLOAT32);
  PyArrayObject* result_paths = (PyArrayObject*) PyArray_SimpleNew(3, dims_paths, NPY_INT32);
  npy_float32 *cost_ptr;
  npy_int32 *path_xptr;
  npy_int32 *path_yptr;
  for (int i = 0; i < h * w; i++) {
    int x = i % w;
    int y = i / w;
    cost_ptr = (npy_float32*) (result_costs->data + y * result_costs->strides[0] + x * result_costs->strides[1]);
    *cost_ptr = (npy_float32) costs[i];
    path_yptr = (npy_int32*) (result_paths->data + y * result_paths->strides[0] + x * result_paths->strides[1]);
    path_xptr = (npy_int32*) (result_paths->data + y * result_paths->strides[0] + x * result_paths->strides[1] + result_paths->strides[2]);
    if (paths[i] < 0) {
        *path_yptr = -1;
        *path_xptr = -1;
    } else {
        *path_yptr = paths[i] / w;
        *path_xptr = paths[i] % w;

    }
  }
  PyObject *result = PyTuple_New(2);
  PyTuple_SetItem(result, 0, (PyObject*) result_costs);
  PyTuple_SetItem(result, 1, (PyObject*) result_paths);

  delete[] costs;
  delete[] nbrs;
  delete[] paths;

  return result;
}

static PyMethodDef dijkstra_methods[] = {
    {"dijkstra", (PyCFunction)dijkstra, METH_VARARGS, "dijkstra"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef dijkstra_module = {
    PyModuleDef_HEAD_INIT,"dijkstra", NULL, -1, dijkstra_methods
};

PyMODINIT_FUNC PyInit_dijkstra(void) {
  import_array();
  return PyModule_Create(&dijkstra_module);
}
