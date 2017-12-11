from mxnet import ndarray as nd
from mxnet import autograd
from mxnet import gluon
import random
import matplotlib as mlp
mlp.rcParams['figure.dpi'] = 120 # for what?
import matplotlib.pyplot as plt


num_train = 20 # 训练样本数
num_test = 100 # 测试样本数
num_inputs = 200 # 样本维度

true_w = nd.ones((num_inputs, 1)) * 0.01
true_b = 0.05

X = nd.random_normal(shape=(num_train + num_test, num_inputs))
y = nd.dot(X, true_w)
y += 0.01 * nd.random_normal(shape=y.shape)

X_train, X_test = X[:num_train, :], X[num_train:, :] # 语法?
y_train, y_test = y[:num_train], y[num_train:]

# 当我们开始训练神经网络的时候，我们需要不断读取数据块。
# 这里我们定义一个函数它每次返回batch_size个随机的样本和对应的目标。
# 我们通过python的yield来构造一个迭代器。
batch_size = 1
def data_iter(num_examples):
    idx = list(range(num_examples))
    random.shuffle(idx)
    for i in range(0, num_examples, batch_size):
        j = nd.array(idx[i:min(i+batch_size, num_examples)])
        yield X.take(j), y.take(j)

# 初始化模型参数
def get_params():
    w = nd.random_normal(shape=(num_inputs, 1)) * 0.01
    b = nd.zeros((1,))
    for param in (w, b):
        param.attach_grad()
    return (w, b)

# 正则化
def L2_penalty(w, b):
    return (w**2).sum() + b**2

# 训练和测试
def net(X, lambd, w, b):
    return nd.dot(X, w) + b

def square_loss(yhat, y):
    return (yhat - y.reshape(yhat.shape)) ** 2

def SGD(params, lr):
    for param in params:
        param[:] = param - lr * param.grad

def mytest(parms, X, y):
    return square_loss(net(X, 0, *parms), y).mean().asscalar()

def train(lambd):
    epochs = 10
    learning_rate = 0.002
    params = get_params()
    train_loss = []
    test_loss = []
    for e in range(epochs):
        for data, label in data_iter(num_train):
            with autograd.record():
                output = net(data, lambd, *params) # 语法?
                loss = square_loss(output, label) + lambd * L2_penalty(*params)
            loss.backward()
            SGD(params, learning_rate)
        train_loss.append(mytest(params, X_train, y_train))
        test_loss.append(mytest(params, X_test, y_test))
    plt.plot(train_loss)
    plt.plot(test_loss)
    plt.legend(['train','test'])
    plt.show()
    return 'learned w[:10]:',params[0][:10],'learned b:',params[1]

# 未正则化
# train(0)

# 正则化
print(train(2))



