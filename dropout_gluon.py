from mxnet.gluon import nn
from mxnet import nd
from mxnet import autograd
from mxnet import gluon
import utils

# 定义模型
net = nn.Sequential()
drop_prob1 = 0.3
drop_prob2 = 0.6
with net.name_scope():
    net.add(nn.Flatten())
    # 第一层全连接
    net.add(nn.Dense(256, activation="relu"))
    net.add(nn.Dropout(drop_prob1))

    net.add(nn.Dense(256, activation="relu"))
    net.add(nn.Dropout(drop_prob2))

    net.add(nn.Dense(10))
net.initialize()

# 读取数据并训练
batch_size = 256
train_data, test_data = utils.load_data_fashion_mnist(batch_size)

softmax_cross_entroy = gluon.loss.SoftmaxCrossEntropyLoss()
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate':0.5})

epochs = 5
for epoch in range(epochs):
    train_loss = 0.
    train_acc = 0.
    for data, label in train_data:
        with autograd.record():
            output = net(data)
            loss = softmax_cross_entroy(output, label)
        loss.backward()
        trainer.step(batch_size)

        train_loss += nd.mean(loss).asscalar()
        train_acc += utils.accuracy(output, label)
    test_acc = utils.evaluate_accuracy(test_data, net)
    print("Epoch %d. Loss: %f, Train acc %f, Test acc %f" % (
        epoch, train_loss / len(train_data),
        train_acc / len(train_data), test_acc))
