import tensorlib as tl
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Hyperparameters
input_size = 4  # There are 4 features in the Iris dataset
hidden_size = 5
output_size = 1
learning_rate = 0.01
epochs = 2

# Load Iris dataset
iris = load_iris()
X_data = iris.data
y_data = iris.target

# Convert to binary classification: class 0 (Setosa) vs class 1 (Versicolor)
binary_classes = np.where(y_data < 2, 1, 0)  

X_train, X_test, y_train, y_test = train_test_split(X_data, binary_classes, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X = tl.Tensor(X_train.astype(np.float32), device=tl.Device.CPU, requires_grad=False)
y = tl.Tensor(y_train.astype(np.float32).reshape(-1, 1), device=tl.Device.CPU, requires_grad=False)

W1 = tl.randn([input_size, hidden_size], requires_grad=True)
b1 = tl.zeros([hidden_size], requires_grad=True)
W2 = tl.randn([hidden_size, output_size], requires_grad=True)
b2 = tl.zeros([output_size], requires_grad=True)

def sigmoid(x):
    return 1.0 / (1.0 + tl.exp(-x))

def forward(X):
    hidden = tl.relu(X @ W1 + b1)
    output = sigmoid(hidden @ W2 + b2)
    return output

def binary_cross_entropy(pred, target):
    eps = 1e-7  # Avoid log(0)
    return tl.mean(
        -(target * tl.log(pred + eps) + (1 - target) * tl.log(1 - pred + eps))
    )

losses = []

for epoch in range(epochs):
    y_pred = forward(X)
    loss = binary_cross_entropy(y_pred, y)

    print(loss, y_pred)

    loss.backward()

    # Gradient descent
    W1 -= learning_rate * W1.grad
    b1 -= learning_rate * b1.grad
    W2 -= learning_rate * W2.grad
    b2 -= learning_rate * b2.grad

    # Clear gradients
    W1.zero_()
    b1.zero_()
    W2.zero_()
    b2.zero_()

    losses.append(loss.item())
    print(f"Epoch {epoch}, Loss: {loss.item()}")

print("Training completed.")
