#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <tensorlib/autograd.hpp>
#include <tensorlib/gpu_handler.cuh>
#include <tensorlib/grad_mode.hpp>
#include <tensorlib/node.cuh>
#include <tensorlib/tensor.cuh>
#include <tensorlib/utils.hpp>

// Constructor for 1D tensor
Tensor::Tensor(std::vector<float> data, std::vector<size_t> shape,
               Device device, bool requires_grad)
    : device_(device), requires_grad_(requires_grad) {
  // 1d tensor
  if (shape.empty()) {
    shape.push_back(data.size());
  }
  shape_ = shape;

  stride_.resize(shape.size());
  stride_.back() = 1;
  for (int i = shape.size() - 2; i >= 0; --i) {
    stride_[i] = stride_[i + 1] * shape[i + 1];
  }

  size_t size = shape[0] * stride_[0];

  if (data.size() != size) {
    std::cerr << data.size() << " " << size << std::endl;
    throw std::runtime_error("Data size does not match tensor size");
  }

  // Allocate and copy data based on device
  if (device == Device::CPU) {
    data_ = new float[size];
    std::copy(data.begin(), data.end(), data_);
  } else if (device == Device::GPU) {
    data_ = GPUHandler::allocate_and_copy(data.data(), size);
  }

  if (requires_grad) autograd_meta_ = std::make_shared<AutogradMeta>(this);
}

// Constructor for empty tensor with given shape
Tensor::Tensor(std::vector<size_t> shape, Device device, bool requires_grad)
    : device_(device), requires_grad_(requires_grad) {
  if (shape.empty()) {
    throw std::runtime_error("Shape cannot be empty");
  }

  size_t size = 1;
  for (auto& s : shape) {
    size *= s;
    shape_.push_back(s);
  }

  stride_.resize(shape.size());
  stride_.back() = 1;
  for (int i = shape.size() - 2; i >= 0; --i) {
    stride_[i] = stride_[i + 1] * shape[i + 1];
  }

  // Allocate memory based on device
  if (device == Device::CPU) {
    data_ = new float[size]();
    // set to zero
    std::fill(data_, data_ + size, 0);
  } else if (device == Device::GPU) {
    data_ = GPUHandler::allocate_and_zero(size);
  }

  if (requires_grad) autograd_meta_ = std::make_shared<AutogradMeta>(this);
}

// Destructor
Tensor::~Tensor() {
  if (device_ == Device::CPU) {
    delete[] data_;
  } else if (device_ == Device::GPU) {
    GPUHandler::deallocate(data_);
  }
}

// Move tensor data to device, if already on device, do nothing
variable Tensor::to_device(Device device) {
  if (device == device_) return std::shared_ptr<Tensor>(this);

  variable new_tensor =
      std::make_shared<Tensor>(shape_, device, requires_grad_);

  if (device == Device::CPU) {
    GPUHandler::copy_device_to_host(new_tensor->data_, data_, size());
  } else if (device == Device::GPU) {
    GPUHandler::copy_host_to_device(new_tensor->data_, data_, size());
  }

  return new_tensor;
}

// Set requires_grad
void Tensor::set_requires_grad(bool requires_grad) {
  if (requires_grad && !autograd_meta_) {
    autograd_meta_ = std::make_shared<AutogradMeta>(this);
  }
  requires_grad_ = requires_grad;
}

// Get autograd meta
AutogradMeta& Tensor::autograd_meta() const {
  if (!autograd_meta_) throw std::runtime_error("No autograd meta found");
  return *autograd_meta_;
}

// Get gradient tensor
std::shared_ptr<Tensor> Tensor::grad() const {
  if (!requires_grad()) {
    throw std::runtime_error("Gradient called on non-requires-grad tensor");
  }
  return autograd_meta_->grad_;
}

// Set gradient tensor
void Tensor::set_grad(std::shared_ptr<Tensor> grad) {
  if (!requires_grad()) {
    throw std::runtime_error("Set gradient called on non-requires-grad tensor");
  }

  check_tensor_shape(grad, autograd_meta_->grad_);

  if (grad->device() != autograd_meta_->grad_->device()) {
    grad = grad->to_device(autograd_meta_->grad_->device());
  }
  size_t size = grad->size();

  // Copy gradient data
  if (device_ == Device::CPU) {
    std::copy(grad->data(), grad->data() + size, autograd_meta_->grad_->data());
  } else if (device_ == Device::GPU) {
    GPUHandler::copy_device_to_device(autograd_meta_->grad_->data(),
                                      grad->data(), size);
  }
}

// Perform backpropagation
void Tensor::backward(std::shared_ptr<Tensor> grad) {
  if (!requires_grad()) {
    throw std::runtime_error("Backward called on non-requires-grad tensor");
  }
  if (!GradMode::is_enabled()) {
    throw std::runtime_error("Backward called outside of grad mode");
  }
  // Copy gradient data
  set_grad(grad);

  // no backward pass for leaf tensor
  if (!autograd_meta().grad_fn_) return;

  node_list sorted_nodes = topological_sort(autograd_meta().grad_fn_);
  for (auto& node : sorted_nodes) {
    node->apply();
  }
}

void Tensor::zero_() {
  if (device_ == Device::CPU) {
    std::fill(data_, data_ + size(), 0);
  } else if (device_ == Device::GPU) {
    GPUHandler::zero(data_, size());
  }
}

float Tensor::item() const {
  if (size() != 1) {
    throw std::runtime_error("item() only supports tensors with one element");
  }

  float result;
  if (device_ == Device::CPU) {
    result = data_[0];
  } else if (device_ == Device::GPU) {
    GPUHandler::copy_device_to_host(&result, data_, 1);
  }
  return result;
}

void tensor_to_string_recursive(const float* data,
                                const std::vector<size_t>& shape,
                                const std::vector<size_t>& strides, int dim,
                                size_t offset, std::stringstream& result,
                                int indent) {
  std::string padding(indent,
                      ' ');  // Create padding based on indentation level

  if (dim == shape.size() - 1) {
    result << padding + "[";
    for (size_t i = 0; i < shape[dim]; ++i) {
      result << data[offset + i * strides[dim]];
      if (i < shape[dim] - 1) result << ", ";
    }
    result << "]";
    return;
  }

  result << padding + "[\n";
  for (size_t i = 0; i < shape[dim]; ++i) {
    tensor_to_string_recursive(data, shape, strides, dim + 1,
                               offset + i * strides[dim], result, indent + 2);
    if (i < shape[dim] - 1) result << ",\n";
  }
  result << "\n" + padding + "]";
}

std::string Tensor::to_string() const {
  // std::string result = "Tensor([\n";
  std::stringstream result;

  result << std::fixed << std::setprecision(4) << "Tensor(";

  // Allocate temporary CPU array if data is on GPU
  float* temp = nullptr;
  if (device_ == Device::GPU) {
    temp = new float[size()];
    GPUHandler::copy_device_to_host(temp, data_, size());
  } else {
    temp = data_;
  }

  // Use the CPU data for string formatting
  tensor_to_string_recursive(temp, shape_, stride_, 0, 0, result, 2);

  result << ", ";
  if (requires_grad_) {
    if (autograd_meta_->grad_fn_) {
      result << "grad_fn=<" + autograd_meta_->grad_fn_->name() + ">, ";
    } else {
      result << "grad_fn=<GradAccumulate>, ";
    }
  }
  result << "device=";
  result << (device_ == Device::CPU ? "cpu" : "gpu");
  result << ")";
  return result.str();
}
