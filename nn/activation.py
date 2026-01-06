import numpy as np


class Sigmoid:

    def __init__(self):

        # store during forward and use during backward
        self.A

    def forward(self, Z):

        self.A = 1 / (1 + np.exp(-1 * Z))
        return self.A

    def backward(self, dLdA):

        dAdZ = self.A * (1 - self.A)

        dLdZ = dLdA * dAdZ
        return dLdZ


class Tanh:

    def __init__(self):

        self.A

    def forward(self, Z):

        expZ = np.exp(Z)
        expNegZ = np.exp(-1 * Z)

        self.A = (expZ - expNegZ) / (expZ + expNegZ)
        return self.A

    def backward(self, dLdA):

        dAdZ = 1 - (self.A ** 2)

        dLdZ = dLdA * dAdZ
        return dLdZ


class ReLU:

    def __init__(self):

        self.A

    def forward(self, Z):

        self.A = np.maximum(0, Z)
        return self.A

    def backward(self, dLdA):

        dAdZ = (self.A > 0).astype(dLdA.dtype)

        dLdZ = dLdA * dAdZ
        return dLdZ


class GELU:

    def __init__(self):

        self.gaussianPDF = lambda x: np.exp(-x**2 / 2) / np.sqrt(2*np.pi)
        self.gaussianCDF = lambda x: 0.5 * (1 + np.erf(x / np.sqrt(2)))

        self.A
        self.Z

    def forward(self, Z):

        self.Z = Z

        self.A = Z * self.gaussianCDF(Z)
        return self.A

    def backward(self, dLdA):

        gaussianCDF_Z = self.A / self.Z
        dAdZ = gaussianCDF_Z + (self.Z * self.gaussianPDF(self.Z))

        dLdZ = dLdA * dAdZ
        return dLdZ
