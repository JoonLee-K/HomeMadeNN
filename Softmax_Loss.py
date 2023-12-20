from util import *
import time


class softmax_Loss:
    def __init__(self):
        self.loss = None
        self.y, self.label = None, None
        self.num_classes = 10

    def forward(self, x, label):
        self.label = one_hot_encoding(label, self.num_classes)
        self.y = softmax(x)
        self.loss = cee(self.y, self.label)

        return self.loss

    def backward(self, dout=1):
        batchSize = self.label.shape[0]
        dx = (self.y - self.label) / batchSize
        dx *= dout

        return dx
