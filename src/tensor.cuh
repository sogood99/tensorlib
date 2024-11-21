#ifndef TENSOR
#define TENSOR

#include <cublas_v2.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <functional>
#include <iostream>
#include <memory>
#include <vector>

// Error handling for cuBLAS
#define CHECK_CUBLAS_ERROR(err)                        \
  if (err != CUBLAS_STATUS_SUCCESS) {                  \
    std::cerr << "cuBLAS error: " << err << std::endl; \
    return;                                            \
  }

namespace py = pybind11;

class Tensor {
 private:
  float* data = nullptr;
  std::vector<size_t> mshape;
  size_t msize = 1;
  Node* node = nullptr;

 public:
  Tensor(float* data, std::vector<size_t> shape);
  Tensor(float* data, std::vector<size_t> shape, Node* node);
  Tensor(std::vector<size_t> shape);
  Tensor(py::array_t<float> array);
  ~Tensor();

  void set_node(Node* node);

  static Tensor add(Tensor a, Tensor b);

  py::array_t<size_t> shape();
};

// Node for the computation graph
class Node {
 public:
  std::function<void()> grad_fn;
  std::vector<Tensor*> parents;
  Tensor* tensor = nullptr;

  Node(std::vector<Tensor*> parents, std::function<void()> grad_fn,
       Tensor* tensor);
};
#endif