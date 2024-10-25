import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Generate the dataset
np.random.seed(42)
n = 100
kids_data = np.column_stack(
    (np.random.uniform(30, 45, 50), np.random.uniform(125, 145, 50)))
adults_data = np.column_stack(
    (np.random.uniform(55, 70, 50), np.random.uniform(155, 180, 50)))
data = np.vstack((kids_data, adults_data))
labels = np.hstack((-np.ones(50), np.ones(50)))

# Add a bias term to the data
data_with_bias = np.column_stack((data, np.ones(n)))

# Initialize parameters
theta = np.zeros(3)

# Define the sigmoid function


def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Define the loss function


def loss(theta, X, y):
    z = np.dot(X, theta)
    h = sigmoid(y * z)
    return np.mean(np.log(1 + np.exp(-h)))

# Define the gradient of the loss function


def gradient(theta, X, y):
    z = np.dot(X, theta)
    h = sigmoid(y * z)
    error = h - 1
    gradient_value = np.mean(error[:, np.newaxis] * X, axis=0)
    return gradient_value


# Gradient Descent
def gradient_descent(X, y, alpha, T):
    theta = np.zeros(3)
    loss_history = []

    for _ in range(T):
        gradient_value = gradient(theta, X, y)
        theta -= alpha * gradient_value
        loss_value = loss(theta, X, y)
        loss_history.append(loss_value)

    return theta, loss_history

# Stochastic Gradient Descent


def stochastic_gradient_descent(X, y, alpha, T, S):
    theta = np.zeros(3)
    loss_history = []

    for _ in range(T):
        idx = np.random.choice(len(X), S, replace=False)
        X_batch, y_batch = X[idx], y[idx]

        gradient_value = gradient(theta, X_batch, y_batch)
        theta -= alpha * gradient_value
        loss_value = loss(theta, X, y)
        loss_history.append(loss_value)

    return theta, loss_history

# Animation function


def update_plot(num, line, data, theta_values):
    line.set_ydata((-theta_values[num] * data[:, 0] - theta_values[2]) / theta_values[1])
    return line,

print(data_with_bias.shape)



# Plotting
fig, ax = plt.subplots()
ax.scatter(kids_data[:, 0], kids_data[:, 1], color='blue', label='Kids')
ax.scatter(adults_data[:, 0], adults_data[:, 1],
           color='orange', label='Adults')

line, = ax.plot([], [], lw=2)
ax.set_xlim(20, 80)
ax.set_ylim(100, 200)
ax.legend()

theta_values_gd, _ = gradient_descent(
    data_with_bias, labels, alpha=0.1, T=1000)
theta_values_sgd, _ = stochastic_gradient_descent(
    data_with_bias, labels, alpha=0.1, T=1000, S=10)

ani_gd = animation.FuncAnimation(fig, update_plot, frames=len(
    theta_values_gd), fargs=(line, data, theta_values_gd), interval=50, repeat=False)
ani_sgd = animation.FuncAnimation(fig, update_plot, frames=len(
    theta_values_sgd), fargs=(line, data, theta_values_sgd), interval=50, repeat=False)

plt.show()
