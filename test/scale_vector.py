import tensorlib
import numpy as np

# Test with a sample vector
x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
y = tensorlib.Tensor(x)

print("Vector x:", y.shape())
