import math

from matplotlib import pyplot as plt
from Affine import affine
from ReLU import relu
from Softmax_Loss import softmax_Loss
from load_mnist import loadData
import numpy as np
from util import softmax


np.random.seed(123)
'''
The shape of the training set feature matrix is: (60000, 784)
The shape of the training label vector is: (60000,)
The shape of the test set feature matrix is: (10000, 784)
The shape of the test label vector is: (10000,)
'''
x_train, y_train, x_test, y_test = loadData()

x_train_size = x_train.shape[0]
batch = 32  # use this!!

W1 = np.random.normal(size=(x_train.shape[1], 500))
b1 = np.zeros(500)

W2 = np.random.normal(size=(500, 250))  # data 바뀌면 다르게 처리
b2 = np.zeros(250)

W3 = np.random.normal(size=(250, 10))  # data 바뀌면 다르게 처리
b3 = np.zeros(10)

af1 = affine(W1, b1)
relu1 = relu()

af2 = affine(W2, b2)
relu2 = relu()

af3 = affine(W3, b3)

softmaxLoss = softmax_Loss()

lr = 1e-4
lossList = list()
accList = list()
epoch = 50


def accuracy():
    count = 0
    # predict
    for d, l in zip(x_test, y_test):
        # x -> train data
        # y -> label data

        # forwarding
        x = af1.forward(d)
        x = relu1.forward(x)
        x = af2.forward(x)
        x = relu2.forward(x)
        x = af3.forward(x)
        ans = np.argmax(x)

        if ans == l:
            count += 1

    return count / len(y_test) * 100.0


train_size = x_train.shape[0]
iter_per_epoch = int(math.ceil(train_size / batch)) # 한 번 돌 때 사이즈
# train
for e in range(1, epoch + 1):
    lossEpoch = list()
    for _ in range(iter_per_epoch):
        batch_mask = np.random.choice(train_size, batch)  # 데이터를 랜덤하게 뽑음
        x_batch = x_train[batch_mask]  # 문제
        y_batch = y_train[batch_mask]  # 정답

        if x_batch.ndim == 1:
            y_batch = y_batch.reshape(1, y_batch.size)
            x_batch = x_batch.reshape(1, x_batch.size)

        x = af1.forward(x_batch)
        x = relu1.forward(x)
        x = af2.forward(x)
        x = relu2.forward(x)
        x = af3.forward(x)
        loss = softmaxLoss.forward(x, y_batch)
        lossEpoch.append(loss)

        dout = softmaxLoss.backward()
        dout = af3.backward(dout=dout, lr=lr)
        dout = relu2.backward(dout)
        dout = af2.backward(dout=dout, lr=lr)
        dout = relu1.backward(dout)
        dout = af1.backward(dout=dout, lr=lr)

    lossAvg = sum(lossEpoch) / len(lossEpoch)
    print(f'epoch {e} : {lossAvg}')
    lossList.append(lossAvg)

    if e % 10 == 0:
        acc = accuracy()
        print(f'accuracy : {acc}%')
        accList.append(acc)


plt.plot(lossList)
plt.show()

plt.plot(accList)
plt.show()



