import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import sys


class Solutions:
    def __init__(self) -> None:
        self.x_star = np.array([1, 1])
        self.T = 10000

    @staticmethod
    def f(x):
        return 100*(x[1]-(x[0]**2))**2 + (x[0]-1)**2

    @staticmethod
    def grad(x):
        return np.array([400*x[0]**3 + (2-400*x[1])*x[0] - 2, 200*(x[1] - x[0]**2)])

    @staticmethod
    def hessian(x):
        return np.array([[1200*x[0]**2 - 400*x[1]+2, -400*x[0]], [-400*x[0], 200]])

    @staticmethod
    def gradient_descent(x, alpha):
        grads = Solutions.grad(x)
        return x - alpha*grads

    @staticmethod
    def gradient_descent_heavy_ball(x, x_prev, alpha, beta=0.9):
        grads = Solutions.grad(x)
        return x + beta*(x - x_prev) - alpha*grads

    @staticmethod
    def nesterov_accelerated_gradient(x, x_prev, alpha, beta=0.9):
        grads = Solutions.grad(x + beta*(x - x_prev))
        return x + beta*(x - x_prev) - alpha*grads

    @staticmethod
    def newton_method(x):
        grads = Solutions.grad(x).reshape((-1, 1))
        hess = np.linalg.inv(Solutions.hessian(x))
        return x - (hess@grads).reshape((-1,))

    @staticmethod
    def quasi_newton(x, alpha):
        grads = Solutions.grad(x).reshape((2, 1))
        hess = np.linalg.inv(Solutions.hessian(x))
        return x - alpha*(hess*np.eye(hess.shape[0]))@grads.reshape((-1,))

    @staticmethod
    def rms_prop(x, v, alpha, beta=0.9, eps=1e-8):
        grads = Solutions.grad(x)
        v = beta*v + (1-beta)*(grads**2)
        return (x - (alpha/np.sqrt(v+eps))*grads, v)

    @staticmethod
    def adam(x, v, m, i, alpha, eps=0.1, beta1=0.9, beta2=0.9):
        grads = Solutions.grad(x)
        m = beta1*m + (1-beta1)*grads
        v = beta2*v + (1-beta2)*(grads**2)
        m_hat = m/(1-beta1**i)
        v_hat = v/(1-beta2**i)
        return (x - (alpha*m_hat)/(np.sqrt(v_hat)+eps), v, m)

    def exercise1(self):
        self.exercise1a()

    def plot_gradient_descent(self):
        x_start = np.array([-2, 2])
        alpha = 0.001
        x_s = []
        f_s = []

        for i in range(self.T):
            x_s.append(x_start)
            f_s.append(Solutions.f(x_start))
            x_start = Solutions.gradient_descent(x_start, alpha)

        # 2D plot
        norm_distances = np.sqrt(np.sum(np.square(x_s - x_start), axis=1))
        f_differences = np.abs(f_s)

        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].plot(range(self.T), norm_distances)
        ax[0].set(xlabel='t', ylabel="x_distance")
        ax[0].set_title('Measure of X vs t')
        ax[1].plot(range(self.T), f_differences)
        ax[1].set(xlabel='t', ylabel="f_distance")
        ax[1].set_title('Measure of f vs t')
        plt.show()

        # 3D plot
        X = np.linspace(-2.5, 2.5, 100)
        Y = np.linspace(-2.5, 2.5, 100)
        X, Y = np.meshgrid(X, Y)

        x_s = np.array(x_s)

        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

        Z = [Solutions.f([x, y]) for x, y in zip(X.flatten(), Y.flatten())]
        Z = np.array(Z).reshape(X.shape)

        surf = ax.plot_wireframe(X, Y, Z, rstride=10, cstride=10)
        fig.colorbar(surf, shrink=0.5, aspect=5)
        ax.scatter(x_s[:, 0], x_s[:, 1], f_s)
        ax.set_xlabel('x[1]')
        ax.set_ylabel('x[2]')
        ax.set_zlabel('f')
        plt.show()

    def plot_heavy_ball_method(self):
        x_start = np.array([-2, 2])
        x_prev = np.array([-2, 2])
        alpha = 0.001
        x_s = []
        f_s = []

        for i in range(self.T):
            x_s.append(x_start)
            f_s.append(Solutions.f(x_start))
            x_start = Solutions.gradient_descent_heavy_ball(
                x_start, x_prev, alpha)
            x_prev = x_start

        # 2D plot
        norm_distances = np.sqrt(np.sum(np.square(x_s - x_start), axis=1))
        f_differences = np.abs(f_s)

        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].plot(range(self.T), norm_distances)
        ax[0].set(xlabel='t', ylabel="x_distance")
        ax[0].set_title('Measure of X vs t')
        ax[1].plot(range(self.T), f_differences)
        ax[1].set(xlabel='t', ylabel="f_distance")
        ax[1].set_title('Measure of f vs t')
        plt.show()

        # 3D plot
        X = np.linspace(-2.5, 2.5, 100)
        Y = np.linspace(-2.5, 2.5, 100)
        X, Y = np.meshgrid(X, Y)

        x_s = np.array(x_s)

        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

        Z = [Solutions.f([x, y]) for x, y in zip(X.flatten(), Y.flatten())]
        Z = np.array(Z).reshape(X.shape)

        surf = ax.plot_wireframe(X, Y, Z, rstride=10, cstride=10)
        fig.colorbar(surf, shrink=0.5, aspect=5)
        ax.scatter(x_s[:, 0], x_s[:, 1], f_s)
        ax.set_xlabel('x[1]')
        ax.set_ylabel('x[2]')
        ax.set_zlabel('f')
        plt.show()

    def plot_nesterov_accelerated_gradient(self):
        x_start = np.array([-2, 2])
        x_prev = np.array([-2, 2])
        alpha = 0.001
        x_s = []
        f_s = []

        for i in range(self.T):
            x_s.append(x_start)
            f_s.append(Solutions.f(x_start))
            x_start = Solutions.nesterov_accelerated_gradient(
                x_start, x_prev, alpha)
            x_prev = x_start

        # 2D plot
        norm_distances = np.sqrt(np.sum(np.square(x_s - x_start), axis=1))
        f_differences = np.abs(f_s)

        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].plot(range(self.T), norm_distances)
        ax[0].set(xlabel='t', ylabel="x_distance")
        ax[0].set_title('Measure of X vs t')
        ax[1].plot(range(self.T), f_differences)
        ax[1].set(xlabel='t', ylabel="f_distance")
        ax[1].set_title('Measure of f vs t')
        plt.show()

        # 3D plot
        X = np.linspace(-2.5, 2.5, 100)
        Y = np.linspace(-2.5, 2.5, 100)
        X, Y = np.meshgrid(X, Y)

        x_s = np.array(x_s)

        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

        Z = [Solutions.f([x, y]) for x, y in zip(X.flatten(), Y.flatten())]
        Z = np.array(Z).reshape(X.shape)

        surf = ax.plot_wireframe(X, Y, Z, rstride=10, cstride=10)
        fig.colorbar(surf, shrink=0.5, aspect=5)
        ax.scatter(x_s[:, 0], x_s[:, 1], f_s)
        ax.set_xlabel('x[1]')
        ax.set_ylabel('x[2]')
        ax.set_zlabel('f')
        plt.show()

    def plot_newton_method(self):
        x_start = np.array([-2, 2])
        x_s = []
        f_s = []

        for i in range(self.T):
            x_s.append(x_start)
            f_s.append(Solutions.f(x_start))
            x_start = Solutions.newton_method(
                x_start)

        # 2D plot
        norm_distances = np.sqrt(np.sum(np.square(x_s - x_start), axis=1))
        f_differences = np.abs(f_s)

        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].plot(range(self.T), norm_distances)
        ax[0].set(xlabel='t', ylabel="x_distance")
        ax[0].set_title('Measure of X vs t')
        ax[1].plot(range(self.T), f_differences)
        ax[1].set(xlabel='t', ylabel="f_distance")
        ax[1].set_title('Measure of f vs t')
        plt.show()

        # 3D plot
        X = np.linspace(-2.5, 2.5, 100)
        Y = np.linspace(-2.5, 2.5, 100)
        X, Y = np.meshgrid(X, Y)

        x_s = np.array(x_s)

        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

        Z = [Solutions.f([x, y]) for x, y in zip(X.flatten(), Y.flatten())]
        Z = np.array(Z).reshape(X.shape)

        surf = ax.plot_wireframe(X, Y, Z, rstride=10, cstride=10)
        fig.colorbar(surf, shrink=0.5, aspect=5)
        ax.scatter(x_s[:, 0], x_s[:, 1], f_s)
        ax.set_xlabel('x[1]')
        ax.set_ylabel('x[2]')
        ax.set_zlabel('f')
        plt.show()

    def plot_quasi_newton_method(self):
        x_start = np.array([-2, 2])
        alpha = 0.001
        x_s = []
        f_s = []

        for i in range(self.T):
            x_s.append(x_start)
            f_s.append(Solutions.f(x_start))
            x_start = Solutions.quasi_newton(x_start, alpha)

        # 2D plot
        norm_distances = np.sqrt(np.sum(np.square(x_s - x_start), axis=1))
        f_differences = np.abs(f_s)

        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].plot(range(self.T), norm_distances)
        ax[0].set(xlabel='t', ylabel="x_distance")
        ax[0].set_title('Measure of X vs t')
        ax[1].plot(range(self.T), f_differences)
        ax[1].set(xlabel='t', ylabel="f_distance")
        ax[1].set_title('Measure of f vs t')
        plt.show()

        # 3D plot
        X = np.linspace(-2.5, 2.5, 100)
        Y = np.linspace(-2.5, 2.5, 100)
        X, Y = np.meshgrid(X, Y)

        x_s = np.array(x_s)

        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

        Z = [Solutions.f([x, y]) for x, y in zip(X.flatten(), Y.flatten())]
        Z = np.array(Z).reshape(X.shape)

        surf = ax.plot_wireframe(X, Y, Z, rstride=10, cstride=10)
        fig.colorbar(surf, shrink=0.5, aspect=5)
        ax.scatter(x_s[:, 0], x_s[:, 1], f_s)
        ax.set_xlabel('x[1]')
        ax.set_ylabel('x[2]')
        ax.set_zlabel('f')
        plt.show()

    def plot_rms_prop(self):
        x_start = np.array([-2, 2])
        v = np.array([0, 0])
        alpha = 0.001
        x_s = []
        f_s = []

        for i in range(self.T):
            x_s.append(x_start)
            f_s.append(Solutions.f(x_start))
            x_start, v = Solutions.rms_prop(x_start, v, alpha)

        # 2D plot
        norm_distances = np.sqrt(np.sum(np.square(x_s - x_start), axis=1))
        f_differences = np.abs(f_s)

        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].plot(range(self.T), norm_distances)
        ax[0].set(xlabel='t', ylabel="x_distance")
        ax[0].set_title('Measure of X vs t')
        ax[1].plot(range(self.T), f_differences)
        ax[1].set(xlabel='t', ylabel="f_distance")
        ax[1].set_title('Measure of f vs t')
        plt.show()

        # 3D plot
        X = np.linspace(-2.5, 2.5, 100)
        Y = np.linspace(-2.5, 2.5, 100)
        X, Y = np.meshgrid(X, Y)

        x_s = np.array(x_s)

        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

        Z = [Solutions.f([x, y]) for x, y in zip(X.flatten(), Y.flatten())]
        Z = np.array(Z).reshape(X.shape)

        surf = ax.plot_wireframe(X, Y, Z, rstride=10, cstride=10)
        fig.colorbar(surf, shrink=0.5, aspect=5)
        ax.scatter(x_s[:, 0], x_s[:, 1], f_s)
        ax.set_xlabel('x[1]')
        ax.set_ylabel('x[2]')
        ax.set_zlabel('f')
        plt.show()

    def plot_adam(self):
        x_start = np.array([-2, 2])
        v = np.array([0, 0])
        m = np.array([0, 0])
        alpha = 0.1
        x_s = []
        f_s = []

        for i in range(self.T):
            x_s.append(x_start)
            f_s.append(Solutions.f(x_start))
            x_start, v, m = Solutions.adam(x_start, v, m, i+1, alpha)

        # 2D plot
        norm_distances = np.sqrt(np.sum(np.square(x_s - x_start), axis=1))
        f_differences = np.abs(f_s)

        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].plot(range(self.T), norm_distances)
        ax[0].set(xlabel='t', ylabel="x_distance")
        ax[0].set_title('Measure of X vs t')
        ax[1].plot(range(self.T), f_differences)
        ax[1].set(xlabel='t', ylabel="f_distance")
        ax[1].set_title('Measure of f vs t')
        plt.show()

        # 3D plot
        X = np.linspace(-2.5, 2.5, 100)
        Y = np.linspace(-2.5, 2.5, 100)
        X, Y = np.meshgrid(X, Y)

        x_s = np.array(x_s)

        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

        Z = [Solutions.f([x, y]) for x, y in zip(X.flatten(), Y.flatten())]
        Z = np.array(Z).reshape(X.shape)

        surf = ax.plot_wireframe(X, Y, Z, rstride=10, cstride=10)
        fig.colorbar(surf, shrink=0.5, aspect=5)
        ax.scatter(x_s[:, 0], x_s[:, 1], f_s)
        ax.set_xlabel('x[1]')
        ax.set_ylabel('x[2]')
        ax.set_zlabel('f')
        plt.show()

    def exercise1a(self):
        self.plot_gradient_descent()
        self.plot_heavy_ball_method()
        self.plot_nesterov_accelerated_gradient()
        self.plot_newton_method()
        self.plot_quasi_newton_method()
        self.plot_rms_prop()
        self.plot_adam()


if __name__ == '__main__':
    answer = Solutions()
    answer.exercise1()
