from mxnet import ndarray as nd
from mxnet import autograd
from mxnet import gluon
import random
import matplotlib as mlp
mlp.rcParams['figure.dpi'] = 120 # for what?
import matplotlib.pyplot as plt

# 高维线性回归数据集
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

# 训练和数据集
batch_size = 1
dataset_train = gluon.data.ArrayDataset(X_train, y_train)
data_iter_train = gluon.data.DataLoader(dataset_train, batch_size, shuffle=True)

square_loss = gluon.loss.L2Loss()

def mytest(net, X, y):
    return square_loss(net(X), y).mean().asscalar()

def train(weight_decay):
    learning_rate = 0.005
    epochs = 10

    net = gluon.nn.Sequential() # 创建网络
    with net.name_scope():
        net.add(gluon.nn.Dense(1))
    net.initialize()

    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate':learning_rate,'wd':weight_decay})
    train_loss = []
    test_loss = []

    for e in range(epochs):
        for data, label in data_iter_train:
            with autograd.record():
                output = net(data)
                loss = square_loss(output, label)
            loss.backward()
            trainer.step(batch_size)
        train_loss.append(mytest(net, X_train, y_train))
        test_loss.append(mytest(net, X_train, y_test))

    plt.plot(train_loss)
    plt.plot(test_loss)
    plt.legend(['train','test'])
    plt.show()

    return ('learned w[:10]:', net[0].weight.data()[:, :10],
            'learned b:', net[0].bias.data())

train(5)









