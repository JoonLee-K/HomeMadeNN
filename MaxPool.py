import numpy as np

class maxpool:
    def __init__(self, stride=1, kernel=3, padding=0):
        self.maxUnit = None  # input 어디가 max였는지 기억
        self.stride = stride
        self.kernel = kernel
        self.padding = padding

    def forward(self, x):
        # shape of x (784,)
        maxFrame = list()
        out = list()
        for i in range(0, x.shape[0], self.stride):
            frame = x[i:self.kernel]
            index = np.argmax(frame)
            maxFrame.append(index)
            out.append(frame[index])

        return np.array(out)

    def backward(self, dout):
        pass


