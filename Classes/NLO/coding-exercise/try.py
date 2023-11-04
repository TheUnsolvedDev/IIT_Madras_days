import numpy as np

# Sample data (replace these with your dataset)
a = np.array([0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5])
b = np.array([2, 1, 4, 3, 6, 5, 8, 7, 10, 9])

# Learning rate for gradient descent
learning_rate = 0.01

# Maximum number of iterations
max_iterations = 1000

# Initial guess for parameters x(1) and x(2), and bias term x(3)
x = np.array([0.0, 0.0])

# Gradient Descent with positivity constraints
for _ in range(max_iterations):
    error = np.dot(a.reshape((-1, 1)),x[:1]) + x[1] - b
    positive_error_indices = error >= 0
    gradient_x1 = 2 * np.sum(a[positive_error_indices] * error[positive_error_indices])
    # gradient_x2 = 2 * np.sum(a[positive_error_indices, 1] * error[positive_error_indices])
    gradient_x3 = 2 * np.sum(error)
    
    # Update parameters with positivity constraints
    x[0] -= learning_rate * gradient_x1
    # x[1] -= learning_rate * gradient_x2
    x[2] -= learning_rate * gradient_x3

# Print the optimized parameters
print("Optimized Parameters (x(1), x(2), x(3)): ", x)