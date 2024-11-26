#ifndef OPERATORS_HPP
#define OPERATORS_HPP

#include <tensorlib/types.hpp>

class Tensor;

variable operator+(const variable& x, const variable& y);
variable operator-(const variable& x, const variable& y);
variable operator*(const variable& x, const variable& y);
variable operator/(const variable& x, const variable& y);
variable matmul(const variable& x, const variable& y);
variable transpose(const variable& x);
variable log(const variable& x);
variable exp(const variable& x);
variable sin(const variable& x);
variable cos(const variable& x);
variable relu(const variable& x);
variable select_idx(const variable& x, size_t index);
variable reshape(const variable& x, std::vector<size_t> shape);
variable flatten(const variable& x);
variable sum(const variable& x, size_t axis);
variable mean(const variable& x, size_t axis);
variable max(const variable& x, size_t axis);
variable min(const variable& x, size_t axis);

#endif