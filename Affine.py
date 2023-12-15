import numpy as np

class affine:
    def __init__(self, W, b):
        # assume that np.dot(x, W) + b
        self.x = None
        self.dW = None
        self.db = None
        self.W = W
        self.b = b

    def forward(self, x):
        self.x = x
        out = np.dot(self.W.T, self.x) + self.b

        return out

    def backward(self, dout):
        dx = np.dot(self.W, dout)
        self.dW = np.dot(self.x, dout)
        self.db = np.sum(dout, axis=0)

        return dx
