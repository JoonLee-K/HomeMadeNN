class relu:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        # assume that type of 'x' is numpy.array
        self.mask = (x < 0)
        out = x[self.mask] = 0
        return out

    def backward(self, dout):
        dx = dout
        dx[self.mask] = 0
        return dx
