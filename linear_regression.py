from mxnet import ndarray as nd
from mxnet import autograd
import numpy as np
import matplotlib.pyplot as plt
import random

num_input = 2 # 每个样本的特征数
num_example = 1000 # 样本数量

true_w = [2,-3.4]
true_b = 4.2

X = nd.random_normal(shape=(num_example,num_input)) # 随机生成数据集
y = true_w[0]*X[:,0] + true_w[1]*X[:,1] + true_b
y += 0.01 * nd.random_normal(shape=y.shape)

#plt.scatter(X[:,1].asnumpy(),y.asnumpy())
#plt.show()


# 数据读取，data_iter函数返回batch_size个随机的样本和对应的目标
# 由yield构造一个迭代器
batch_size = 10
def data_iter():
    # 产生一个随机索引
    idx = list(range(num_example))
    random.shuffle(idx)
    # 遍历所有样本
    for i in range(0, num_example, batch_size):
        j = nd.array(idx[i:min(i + batch_size, num_example)])
        yield nd.take(X,j),nd.take(y,j) #yield?构造迭代器

# 初始化模型化参数
w = nd.random_normal(shape=(num_input, 1))
b = nd.zeros(1,)
params = [w,b]

# 之后训练时我们需要对这些参数求导来更新它们的值，使损失尽量减小；
# 因此我们需要创建它们的梯度。
for param in params:
    param.attach_grad()

# 定义模型,即“y = X*w + b”这个线性函数
def net(X):
    return nd.dot(X,w) + b

# 损失函数(平方误差)
def square_loss(yhat,y):
    # 注意这里我们把y变形成yhat的形状来避免矩阵形状的自动转换
    return (yhat - y.reshape(yhat.shape)) ** 2

# 优化(随机梯度下降SGD)
def SGD(params,lr):
    for param in params:
        param[:] = param - lr * param.grad


# 训练
# 使用epochs表示迭代总次数；一次迭代中，我们每次随机读取固定数个数据点，计算梯度并更新模型参数

# 模型函数
def real_fn(X):
    return 2 * X[:,0] - 3.4 * X[:,1] + 4.2

# 绘制损失随训练次数而降低的折线图，以及预测值和真实值的散点图
def plot(losses, X, sample_size=1000):
    xs = list(range(len(losses)))
    f,(fg1,fg2) = plt.subplots(1, 2)
    fg1.set_title('Loss during training')
    fg1.plot(xs, losses, '-r')
    fg2.set_title('Estimated vs real function')
    fg2.plot(X[:sample_size, 1].asnumpy(), net(X[:sample_size,:]).asnumpy(), 'or', label='Estimated')
    fg2.plot(X[:sample_size, 1].asnumpy(), real_fn(X[:sample_size, :]).asnumpy(), '*g', label='Real')
    fg2.legend()
    plt.show()

epcshs = 5
learning_rate = 0.001
niter = 0
losses = []
moving_loss = 0
smoothing_constant = 0.01

#训练
for e in range(epcshs):
    total_loss = 0

    for data, label in data_iter():
        with autograd.record():
            output = net(data)
            loss = square_loss(output, label)
        loss.backward() # 求导
        SGD(params, learning_rate)
        total_loss += nd.sum(loss).asscalar()

        # 记录每读取一个数据点后，损失的移动平均值的变化；
        niter += 1
        curr_loss = nd.mean(loss).asscalar()
        moving_loss = (1 - smoothing_constant) * moving_loss + (smoothing_constant) * curr_loss #??

        # correct the bias from the moving averages
        est_loss = moving_loss / (1-(1-smoothing_constant)**niter)

        if (niter + 1) % 100 == 0:
            losses.append(est_loss)
            print("Epoch %s, batch %s. Moving avg of loss: %s. Average loss: %f" % (e, niter, est_loss, total_loss/num_example))
            plot(losses,X)

