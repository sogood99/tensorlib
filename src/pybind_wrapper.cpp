#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "tensor.cuh"

namespace py = pybind11;

PYBIND11_MODULE(tensorlib, m) {
  py::class_<Tensor>(m, "Tensor")
      .def(py::init<py::array_t<float>, bool>(), py::arg("array"),
           py::arg("requires_grad") = false)
      .def("shape", &Tensor::shape, "Shape of the Tensor");
}