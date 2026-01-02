import numpy as np


class Linear:

    def __init__(self, in_feature, out_feature, debug=False):

        self.W = np.zeros((out_feature, in_feature))
        self.b = np.zeros(out_feature)

        self.debug = debug

    def forward(self, A):

        # store for backpropogation
        self.A = A
        self.N = A.shape[0]

        # compute the affine function
        Z = (A @ self.W.T) + self.b.T  # b is broadcasted

        return Z

    def backward(self, dLdZ):

        # compute batch gradients
        self.dLdW = dLdZ.T @ self.A
        self.dLdb = dLdZ.T @ np.ones((self.N, 1))

        # propogate the derivative
        dLdA = dLdZ @ self.W

        if self.debug:
            self.dLdA = dLdA

        return dLdA
