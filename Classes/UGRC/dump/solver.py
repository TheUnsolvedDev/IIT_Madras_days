import numpy as np
import os
from scipy.optimize import nnls
import sys
import pylops

class Solver:

    def __init__(self, projection, received, dim_X, dim_Y):

        self.projection = projection
        self.received = received
        self.shape = (dim_Y,dim_X)
        rank = np.linalg.matrix_rank(self.projection)
        self.num_variables = None
        self.image = None
        print(f"Rank of matrix:{rank}")
        print(f"Number of unknowns:{dim_X * dim_Y}")

    def solve_gen_tikhonov(self, operator):
        #operator will be passed at the function call
        #pylops.Laplacian(dims=(dim_X,dim_Y),edge=True, weights=(3, 3), dtype="float32")

        #Initial guess can be added later
        #print(init_guess)
        #x0=init_guess

        X = pylops.optimization.leastsquares.regularized_inversion(
            self.projection,
            self.received,
            [operator],
            epsRs=[np.sqrt(0.1)],
            **dict(damp=np.sqrt(1e-4), iter_lim=50, show=0)
        )[0]
        self.image = X.reshape((self.shape))
        return self.image


    def solve_nnls_tikhonov(self, lambda_parameter):

        self.num_variables = self.projection.shape[1]

        PROJ_TKNV = np.concatenate([self.projection,
                                    np.sqrt(lambda_parameter) * np.eye(self.num_variables)
                                    ])
        RECV_TKNV = np.concatenate([self.received, np.zeros(self.num_variables)])
        X, _ = nnls(PROJ_TKNV, RECV_TKNV)
        self.image = X.reshape(self.shape)
        return self.image

    def solve_vanilla_nnls(self):
        X, _ = nnls(self.projection,self.received)
        self.image = X.reshape(self.shape)
        return self.image