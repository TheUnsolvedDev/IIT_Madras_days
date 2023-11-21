import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


class Solutions:
    def __init__(self) -> None:
        pass

    @staticmethod
    def f(x):
        return 100*(x[1]-(x[0]**2))**2 + (x[0]-1)**2

    @staticmethod
    def grad(x):
        return [400*x[0]**3 + (2-400*x[1])*x[0] - 2, 200*(x[1] - x[0]**2)]

    @staticmethod
    def hessian(x):
        return [[1200*x[0]**2 - 400*x[1]+2, -400*x[0]], [-400*x[0], 200]]

    def exercise1(self):
        self.exercise1a()

    def exercise1a(self):
        X = np.linspace(-1, 1, 100)
        Y = np.linspace(-1, 1, 100)
        X, Y = np.meshgrid(X, Y)

        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

        Z = [Solutions.f([x, y]) for x, y in zip(X.flatten(), Y.flatten())]
        Z = np.array(Z).reshape(X.shape)

        surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                               linewidth=0, antialiased=False)
        # fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.show()


if __name__ == '__main__':
    answer = Solutions()
    answer.exercise1()
