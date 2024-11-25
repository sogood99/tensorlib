# TensorLib

- [x] Implement computational graph.
- [x] Implement backprop by extending computational graph.
- [x] CPU implementation for most tensor functions.
- [ ] GPU implementation for most tensor functions.
- [ ] Simple neural network implementation.

## Running Instructions

### Python

First run
```bash
sh build.sh
```
which creates a build directory with the .so file, then set the PythonPath:
```bash
export PYTHONPATH=$(pwd)/build:$PYTHONPATH
```

### C++