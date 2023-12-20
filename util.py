import numpy as np
np.seterr(divide='ignore')


# extract the large number from small numbers make it smaller in hole list.
def softmax(x):
    e = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e / np.sum(e, axis=1, keepdims=True)


# 코드 바꾸자
def cee(y, label):

    if y.ndim == 1:
        label = label.reshape(1, label.size)
        y = y.reshape(1, y.size)

    batchSize = y.shape[0]
    error = -np.sum(label * np.log(y + 1e-7)) / batchSize
    # return -np.sum(t * np.log(y + 1e-9)) / y.shape[0]
    # error = np.clip(error, 1e-7, 1-1e-7)

    return error


def one_hot_encoding(labels, num_classes):
    # This function converts the labels to one-hot encoded format
    return np.eye(num_classes)[labels]