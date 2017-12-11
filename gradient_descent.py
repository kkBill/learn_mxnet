import mxnet as mx
import random
from mxnet import autograd
from mxnet import ndarray as nd
from mxnet import gluon
import matplotlib.pyplot as plt
import numpy as np

# 语法？
mx.random.seed(1)
random.seed(1)

# 生成数据集
num_inputs = 2
num_examples = 1000
true_w = [2, -3.4]
true_b = 4.2

X = nd.random_normal(scale=1, shape=(num_examples,num_inputs))
y = true_w[0]*X[:,0] + true_w[1]*X[:,1] + true_b
y += 0.01 * nd.random_normal(scale=1, shape=y.shape)
dataset = gluon.data.ArrayDataset(X, y)

# 构造迭代器
def data_iter(batch_size):
    idx = list(range(num_examples))
    random.shuffle(idx)

    for batch_i, i in enumerate(range(0, num_examples, batch_size)): # 关键字？
        j = nd.array(idx[i:min(i+batch_size, num_examples)])
        yield batch_i, X.take(j), y.take(j)

# 初始化模型参数
def init_params():
    w = nd.random_normal(scale=1, shape=(num_inputs, 1))
    b = nd.zeros(shape=(1,))
    params = [w,b]
    for param in params:
        param.attach_grad()
    return params

# 线性回归模型
def net(X, w, b):
    return nd.dot(X,w) + b

# 损失函数
def square_loss(yhat, y):
    return (yhat - y.reshape(yhat.shape))**2 / 2

def sgd(params, lr, batch_size):
    for param in params:
        param[:] = param - lr * param.grad / batch_size

def train(batch_size, lr, epochs, period):
    assert period >= batch_size and period % batch_size == 0 # 若不满足该条件，抛出异常
    w, b = init_params()
    total_loss = [np.mean(square_loss(net(X, w, b), y).asnumpy())]
    for epoch in range(1, epochs + 1):
        # 学习率自我衰减
        if epoch > 2:
            lr *= 0.1
        for batch_i, data, label in data_iter(batch_size):
            with autograd.record():
                output = net(data, w, b)
                loss = square_loss(output, label)
            loss.backward()

            sgd([w,b], lr, batch_size)

            if batch_i * batch_size % period == 0:
                total_loss.append(np.mean(square_loss(net(X,w,b),y).asnumpy()))

        print("Batch size %d, Learning rate %f, Epoch %d, loss %.4e" %
              (batch_size, lr, epoch, total_loss[-1])) # 每次都打印total_loss的最后一个(即最新的)

    print('w:',np.reshape(w.asnumpy(), (1, -1)), # 语法
          'b:',b.asnumpy()[0],'\n')

    x_axis = np.linspace(0, epochs, len(total_loss), endpoint=True)
    plt.semilogy(x_axis, total_loss) # Make a plot with log scaling on the y axis.
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.show()

# 当批量大小为1时，即随机梯度下降(SGD)
# train(batch_size=1, lr=0.2, epochs=3, period=10)

# 当批量大小为1000时(batch size == num_example)，即梯度下降(GD)
# train(batch_size=1000, lr=0.2, epochs=3, period=1000)

# 当批量大小为10时，即小批量随机梯度(mini-batch GD)
train(batch_size=10, lr=0.2, epochs=3, period=10)

# 学习率过大，导致不收敛
# train(batch_size=10, lr=5, epochs=3, period=10)

# 学习率过小，导致收敛的比较慢
# train(batch_size=10, lr=0.0002, epochs=3, period=10)





