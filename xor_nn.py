import numpy as np

np.random.seed(42)

# XOR dataset: output is 1 only when exactly one input is 1
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
y = np.array([[0], [1], [1], [0]], dtype=float)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(a):
    # Derivative written in terms of the activation output a = sigmoid(z)
    return a * (1 - a)

# Network: 2 inputs -> 4 hidden neurons -> 1 output
input_size, hidden_size, output_size = 2, 4, 1
W1 = np.random.randn(input_size, hidden_size)
b1 = np.zeros((1, hidden_size))
W2 = np.random.randn(hidden_size, output_size)
b2 = np.zeros((1, output_size))

learning_rate = 1.0
epochs = 10000

for epoch in range(1, epochs + 1):
    # Forward pass
    z1 = X @ W1 + b1
    a1 = sigmoid(z1)
    z2 = a1 @ W2 + b2
    a2 = sigmoid(z2)

    # Mean squared error loss
    loss = np.mean((a2 - y) ** 2)

    # Backward pass (backpropagation)
    m = X.shape[0]
    d2 = (a2 - y) * sigmoid_derivative(a2) * (2 / m)
    dW2 = a1.T @ d2
    db2 = d2.sum(axis=0, keepdims=True)

    d1 = (d2 @ W2.T) * sigmoid_derivative(a1)
    dW1 = X.T @ d1
    db1 = d1.sum(axis=0, keepdims=True)

    # Gradient descent update
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1

    if epoch % 1000 == 0:
        print(f"Epoch {epoch:5d} | Loss: {loss:.6f}")

print("\nFinal predictions:")
for xi, yi, pi in zip(X, y, a2):
    predicted_class = int(pi[0] > 0.5)
    print(f"Input: {xi.astype(int)}  Target: {int(yi[0])}  "
          f"Output: {pi[0]:.4f}  Predicted: {predicted_class}")