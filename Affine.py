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
        dot = np.dot(self.x, self.W)
        out = dot + self.b

        return out

    def backward(self, dout, lr=1e-4):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        self.W -= lr * self.dW
        self.b -= lr * self.db

        return dx

    def getHyperGrad(self):
        return self.dW, self.db

    def updateParam(self, W, b):
        self.W = W
        self.b = b