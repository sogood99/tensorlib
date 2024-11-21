#include "tensor.cuh"

Tensor::Tensor(float* data, std::vector<size_t> shape) {
  mshape = shape;
  msize = 1;
  for (auto dim : shape) {
    msize *= dim;
  }
  this->data = data;
}

Tensor::Tensor(float* data, std::vector<size_t> shape, Node* node) {
  mshape = shape;
  msize = 1;
  for (auto dim : shape) {
    msize *= dim;
  }
  this->data = data;
  this->node = node;
}

Tensor::Tensor(py::array_t<float> array) {
  py::buffer_info info = array.request();
  msize = 1;
  for (auto dim : info.shape) {
    mshape.push_back(dim);
    msize *= dim;
  }

  cudaMalloc(&data, msize * sizeof(float));
}

Tensor::~Tensor() {
  if (data) {
    cudaFree(data);
  }
}

void Tensor::set_node(Node* node) { this->node = node; }

static Tensor add(Tensor a, Tensor b) {
  Tensor c = Tensor(py::array_t<float>(a.shape()));
  return c;
}

py::array_t<size_t> Tensor::shape() {
  return py::array_t<size_t>(mshape.size(), mshape.data());
}

Node::Node(std::vector<Tensor*> parents, std::function<void()> grad_fn,
           Tensor* tensor) {
  this->parents = parents;
  this->grad_fn = grad_fn;
  this->tensor = tensor;
}