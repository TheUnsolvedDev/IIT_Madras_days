import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.ticker import LinearLocator

plt.ion()


class Solutions:
    def __init__(self) -> None:
        pass

    @staticmethod
    def f(x):
        if len(x) != 2:
            raise "Error Input"
        x = np.array(x).reshape(2, 1)

        sub1_term3 = np.array([2, 1]).reshape(2, 1)
        sub2_term3 = np.array([[4, 1], [1, 4]])
        sub1_term4 = np.array([2, -1]).reshape(2, 1)

        term1 = 0.1*(x[0]-1)**4
        term2 = 0.9*(x[1]**4)
        term3 = (x - sub1_term3).T @ sub2_term3 @ (x - sub1_term3)
        term4 = sub1_term4.T @ x

        return (term1 + term2 + term3 + term4).squeeze()

    def exercise1(self):
        X = np.arange(-5, 5, 0.25)
        Y = np.arange(-5, 5, 0.25)
        X, Y = np.meshgrid(X, Y)

        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

        Z = [Solutions.f([x, y]) for x, y in zip(X.flatten(), Y.flatten())]
        Z = np.array(Z).reshape(X.shape)

        surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                               linewidth=0, antialiased=False)
        # fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.show(block=True)

    def exercise2(self):
        dataset = np.array([[0, 2],
                            [0.5, 1],
                            [1, 4],
                            [1.5, 3],
                            [2, 6],
                            [2.5, 5],
                            [3, 8],
                            [3.5, 7],
                            [4, 10],
                            [4.5, 9]
                            ])
        x = dataset[:, 0].reshape(-1, 1)
        y = dataset[:, 1].reshape(-1, 1)

        X = np.hstack([np.ones_like(x), x])
        m = np.linalg.inv(np.dot(X.T, X))@X.T@y
        line = X@m

        Lambda = 100
        # np.linalg.inv(np.dot(X.T, X) + Lambda*np.eye(X.shape[1]))@X.T@y
        m2 = np.array([[2], [2]])
        m3 = np.array([[0], [2]])
        line2 = X@m2
        line3 = X@m3
        plt.scatter(x, y)
        plt.plot(x, line, color='red')
        plt.plot(x, line2, color='green')
        plt.plot(x, line3, color='green')
        plt.xlabel('x[1]')
        plt.ylabel('x[2]')
        plt.show(block=True)


if __name__ == '__main__':
    answer = Solutions()
    answer.exercise2()
