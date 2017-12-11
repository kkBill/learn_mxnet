from mxnet import ndarray as nd
from mxnet import autograd
from mxnet import gluon

num_input = 2 # 每个样本的特征数
num_example = 1000 # 样本数量

true_w = [2,-3.4]
true_b = 4.2

# 创建数据集
X = nd.random_normal(shape=(num_example, num_input))
y = true_w[0] * X[:,0] + true_w[1] * X[:,1] + true_b
y += 0.01 * nd.random_normal(shape=y.shape)

# 数据读取(使用gluon的data模块)
batch_size = 10
dataset = gluon.data.ArrayDataset(X, y)
data_iter = gluon.data.DataLoader(dataset, batch_size, shuffle=True)

# 定义模型
net = gluon.nn.Sequential()
# 加入一个Dense层，它唯一必须定义的参数就是输出节点的个数，在线性模型里面是1.
net.add(gluon.nn.Dense(1))

# 初始化模型参数(这里使用默认随机初始化方法)
net.initialize()


# 损失函数(平方误差函数)
square_loss = gluon.loss.L2Loss()

# 优化(无需手动实现SGD算法，创建一个Trainer实例并将模型参数传递给它即可)
trainer = gluon.Trainer(net.collect_params(),'sgd',{'learning_rate':0.1})

# 训练
epochs = 5
batch_size = 10
for e in range(epochs):
    total_loss = 0
    for data, label in data_iter:
        with autograd.record():
            output = net(data)
            loss = square_loss(output,label)
        loss.backward() # 求导
        trainer.step(batch_size) # 更新模型
        # help(trainer.step)
        total_loss += nd.sum(loss).asscalar()
    print("Epoch %d, average loss:%f" % (e,total_loss/num_example))


