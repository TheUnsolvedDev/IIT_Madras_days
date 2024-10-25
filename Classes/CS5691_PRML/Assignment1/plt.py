import numpy as np
import matplotlib.pyplot as plt


# Function to generate synthetic data for a multivariate Gaussian distribution
def generate_multivariate_gaussian(mean, cov_matrix, num_samples):
    return np.random.multivariate_normal(mean, cov_matrix, num_samples)


# Create synthetic data for eight different distributions
num_samples = 100
distributions = []

for i in range(8):
    mean = np.random.rand(2) * 10  # Random mean vector
    cov_matrix = np.random.rand(2, 2) * 3  # Random covariance matrix
    # Ensure positive semi-definite
    cov_matrix = np.dot(cov_matrix, cov_matrix.T)
    data = generate_multivariate_gaussian(mean, cov_matrix, num_samples)
    distributions.append((mean, cov_matrix, data))

means_all = [{1.0: np.array([3.59338209, 2.68809255]),
           2.0: np.array([12.94573314, 10.49833743]),
           3.0: np.array([0.96583767, 12.75812286])},

          {1.0: np.array([3.59338209, 2.68809255]),
           2.0: np.array([12.94573314, 10.49833743]),
           3.0: np.array([0.96583767, 12.75812286])},
          {1.0: np.array([3.59338209, 2.68809255]),
           2.0: np.array([12.94573314, 10.49833743]),
           3.0: np.array([0.96583767, 12.75812286])},
          {1.0: np.array([3.59338209, 2.68809255]),
           2.0: np.array([12.94573314, 10.49833743]),
           3.0: np.array([0.96583767, 12.75812286])},
          {1.0: np.array([25.80354, 24.31188457]),
           2.0: np.array([25.03616286, 25.07001714]),
           3.0: np.array([5.02284541, 14.86669754])},
          {1.0: np.array([25.80354, 24.31188457]),
           2.0: np.array([25.03616286, 25.07001714]),
           3.0: np.array([5.02284541, 14.86669754])},
          {1.0: np.array([25.80354, 24.31188457]),
           2.0: np.array([25.03616286, 25.07001714]),
           3.0: np.array([5.02284541, 14.86669754])},
          {1.0: np.array([25.80354, 24.31188457]),
           2.0: np.array([25.03616286, 25.07001714]),
           3.0: np.array([5.02284541, 14.86669754])}]

covars_all = [{1.0: np.array([[1.55029091, 0.44332785],
                           [0.44332785, 2.02276141]]),
            2.0: np.array([[1.55029091, 0.44332785],
                           [0.44332785, 2.02276141]]),
            3.0: np.array([[1.55029091, 0.44332785],
                           [0.44332785, 2.02276141]])},
           {1.0: np.array([[1.55029091, 0.44332785],
                           [0.44332785, 2.02276141]]),
            2.0: np.array([[1.20754765, 0.28817679],
                           [0.28817679, 1.39535854]]),
            3.0: np.array([[1.61688054, 0.62863937],
                           [0.62863937, 1.04855286]])},
           {1.0: np.array([[1.54586151, 0.],
                           [0., 2.01698209]]),
            2.0: np.array([[1.54586151, 0.],
                           [0., 2.01698209]]),
            3.0: np.array([[1.54586151, 0.],
                           [0., 2.01698209]])},
           {1.0: np.array([[1.54586151, 0.],
                           [0., 2.01698209]]),
            2.0: np.array([[1.20409751, 0.],
                           [0., 1.3913718]]),
            3.0: np.array([[1.61226088, 0.],
                           [0., 1.045557]])},
           {1.0: np.array([[45.83651718, -3.19956404],
                           [-3.19956404, 50.14754862]]),
            2.0: np.array([[45.83651718, -3.19956404],
                           [-3.19956404, 50.14754862]]),
            3.0: np.array([[45.83651718, -3.19956404],
                           [-3.19956404, 50.14754862]])},
           {1.0: np.array([[45.83651718, -3.19956404],
                           [-3.19956404, 50.14754862]]),
            2.0: np.array([[6.01788109, 0.24697729],
                           [0.24697729, 6.8156035]]),
            3.0: np.array([[23.17833338, -1.57561427],
                           [-1.57561427, 23.35292349]])},
           {1.0: np.array([[45.7055557,  0.],
                           [0., 50.00426991]]),
            2.0: np.array([[45.7055557,  0.],
                           [0., 50.00426991]]),
            3.0: np.array([[45.7055557,  0.],
                           [0., 50.00426991]])},
           {1.0: np.array([[45.7055557,  0.],
                           [0., 50.00426991]]),
            2.0: np.array([[6.00068715, 0.],
                           [0., 6.79613034]]),
            3.0: np.array([[23.11210957,  0.],
                           [0., 23.28620086]])}]

# Create eight 3D plots
fig = plt.figure(figsize=(14, 14))

for i, (mean, cov) in enumerate(zip(means_all,covars_all)):
    ax = fig.add_subplot(2, 2, i + 1, projection='3d')
    x, y = np.meshgrid(np.linspace(-10, 40, 100), np.linspace(-10, 40, 100))
    xy = np.column_stack([x.ravel(), y.ravel()])
    for cls in mean.keys():
        pdf_values = np.zeros(xy.shape[0])
        for j in range(xy.shape[0]):
            diff = xy[j] - mean[cls]
            
            pdf_values[j] = np.exp(-0.5 * np.dot(np.dot(diff.T,
                                np.linalg.inv(cov[cls])), diff))

        pdf_values = pdf_values.reshape(x.shape)

        ax.plot_surface(x, y, pdf_values, cmap='viridis')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('PDF')
    ax.set_title(f'Distribution {i + 1}')

# Adjust spacing for better layout
plt.tight_layout()
plt.show()
