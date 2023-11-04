import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.ticker import LinearLocator


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
        
        Z = [Solutions.f([x,y]) for x,y in zip(X.flatten(),Y.flatten())]
        Z = np.array(Z).reshape(X.shape)
        
        surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
        # fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.show()
        
        


if __name__ == '__main__':
    answer = Solutions()
    answer.exercise1()
