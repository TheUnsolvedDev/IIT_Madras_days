import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the function L(x, y)


def f(x, y):
    return x**2+y**2+x*y


def L(x, y):
    return 49 + 11 * (x - 3) + 13 * (y - 5)


if __name__ == '__main__':
    # Adjust the range and number of points as needed
    x = np.linspace(-10, 10, 100)
    y = np.linspace(-10, 10, 100)
    X, Y = np.meshgrid(x, y)
    Z = L(X, Y)
    Z2 = f(X, Y)

    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the surface
    surf = ax.plot_surface(X, Y, Z, cmap='viridis')
    surf2 = ax.plot_surface(X, Y, Z2)

    # Add labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Plot of 49 + 11(x - 3) + 13(y - 5) and $x^2 + y^2 + xy$')

    # Show the plot
    plt.show()
