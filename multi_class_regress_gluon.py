import utils
from mxnet import gluon
from mxnet import ndarray as nd
from mxnet import autograd

# 获取和读取数据
batch_size = 256
train_data, test_data = utils.load_data_fashion_mnist(batch_size)

# 定义和初始化模型
net = gluon.nn.Sequential()
with net.name_scope():
    net.add(gluon.nn.Flatten())
    net.add(gluon.nn.Dense(10))
net.initialize()

# Softmax和交叉熵损失函数
softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()

# 优化
trainer = gluon.Trainer(net.collect_params(),'sgd',{'learning_rate':0.1})

# 训练
for epoch in range(5):
    train_acc = 0.
    train_loss = 0.
    for data, label in train_data:
        with autograd.record():
            output = net(data)
            loss = softmax_cross_entropy(output, label)
        loss.backward()
        trainer.step(batch_size)

        train_loss += nd.mean(loss).asscalar()
        train_acc += utils.accuracy(output, label)

    test_acc = utils.evaluate_accuracy(test_data, net)
    print("Epoch %d. Loss: %f, Train acc: %f, Test acc: %f" %
          (epoch, train_loss/len(train_data), train_acc/len(train_data), test_acc))
