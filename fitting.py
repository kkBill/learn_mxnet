from mxnet import ndarray as nd
from mxnet import autograd
from mxnet import gluon
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 120 #?

# 数据生成函数 y=1.2x−3.4x^2+5.6x^3+5.0+noise
num_train = 100
num_test = 100
true_w = [1.2, -3.4, 5.6]
true_b = 5.0

# 生成数据集
x = nd.random_normal(shape=(num_train+num_test, 1))
X = nd.concat(x, nd.power(x, 2), nd.power(x, 3))
y = true_w[0] * X[:, 0] + true_w[1] * X[:, 1] + true_w[2] * X[:, 2] + true_b
y += 0.1 * nd.random_normal(0,0.1,shape=y.shape)

def train(X_train, X_test, y_train, y_test):
    # 线性回归模型
    net = gluon.nn.Sequential()
    with net.name_scope():
        net.add(gluon.nn.Dense(1))
    net.initialize()
    # 设一些默认参数
    learning_rate = 0.01
    epochs = 100
    batch_size = min(10, y_train.shape[0])
    dataset_train = gluon.data.ArrayDataset(X_train, y_train)
    data_iter_train = gluon.data.DataLoader(dataset_train, batch_size, shuffle=True)
    # 默认SGD和均方误差
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate':learning_rate})
    square_loss = gluon.loss.L2Loss()
    # 保存训练和测试损失
    train_loss = []
    test_loss = []
    for epoch in range(epochs):
        for data, label in data_iter_train:
            with autograd.record():
                output = net(data)
                loss = square_loss(output, label)
            loss.backward()
            trainer.step(batch_size)
            train_loss.append(square_loss(net(X_train),y_train).mean().asscalar())
            test_loss.append(square_loss(net(X_test),y_test).mean().asscalar())

    # 打印结果
    plt.plot(train_loss)
    plt.plot(test_loss)
    plt.legend(['train','test'])
    plt.show()
    return ('learned weight',net[0].weight.data(),
            'learned bias',net[0].bias.data())

# 正常
#train(X[:num_train,:],X[num_train:,:],y[:num_train],y[num_train:])

# underfitting(线性模型在非线性模型生成的数据集上容易欠拟合。)
#train(x[:num_train,:],x[num_train:,:],y[:num_train],y[num_train:])

# overfitting(训练样本太少)
train(X[:2,:],X[num_train:,:],y[:2],y[num_train:])

print('hello world')
