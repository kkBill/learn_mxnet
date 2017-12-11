from mxnet import gluon
from mxnet import ndarray as nd
from mxnet import autograd
import matplotlib.pyplot as plt

def transform(data, label):
    return data.astype('float32')/255, label.astype('float32')

mnist_train = gluon.data.vision.FashionMNIST(train=True,transform=transform)
mnist_test = gluon.data.vision.FashionMNIST(train=False,transform=transform)

#data, label = mnist_train[0]
#print('example shape: ',data.shape, 'label',label)

def show_images(images):
    n = images.shape[0]
    _, figs = plt.subplots(1, n, figsize=(15,15)) #
    for i in range(n):
        figs[i].imshow(images[i].reshape((28,28)).asnumpy())
        figs[i].axes.get_xaxis().set_visible(False)
        figs[i].axes.get_yaxis().set_visible(False)
    plt.show()

def get_text_labels(label):
    text_label = ['t-shirt', 'trouser', 'pullover', 'dress,', 'coat',
        'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_label[int(i)] for i in label] #???


# 数据读取
batch_size = 256
# shuffle=True表示每次从训练数据里读取一个由随机样本组成的批量
train_data = gluon.data.DataLoader(mnist_train, batch_size, shuffle=True)
test_data = gluon.data.DataLoader(mnist_test, batch_size, shuffle=False)

# 初始化模型参数
num_inputs = 784 # 28*28
num_outputs = 10 # 多分类问题，输出有10类

W = nd.random_normal(shape=(num_inputs, num_outputs))
b = nd.random_normal(shape=num_outputs)
params = [W, b]

# 对模型参数附上梯度
for param in params:
    param.attach_grad()

def softmax(X):
    exp = nd.exp(X)
    # 假设exp是矩阵，这里对行进行求和，并要求保留axis 1，
    # 就是返回 (nrows, 1) 形状的矩阵
    partition = exp.sum(axis=1, keepdims=True)
    return exp/partition

# 定义模型
def net(X):
    return softmax(nd.dot(X.reshape((-1,num_inputs)), W) + b)

# 交叉熵损失函数(针对预测为概率值的损失函数)
def cross_entropy(yhat, y):
    return - nd.pick(nd.log(yhat), y)

# 计算精度
def accuracy(output, label):
    return nd.mean(output.argmax(axis=1)==label).asscalar()

# 评估
def evaluate_accuracy(data_iterator, net):
    acc = 0
    for data, label in data_iterator:
        output = net(data)
        acc += accuracy(output, label)
    return acc / len(data_iterator)

# 优化(随机梯度下降SGD)
def SGD(params,lr):
    for param in params:
        param[:] = param - lr * param.grad

# 训练
learning_rate = 0.1
for epoch in range(5):
    train_loss = 0.
    train_acc = 0.
    for data, label in train_data:
        with autograd.record():
            output = net(data)
            loss = cross_entropy(output, label)
        loss.backward()

        # 将梯度做平均，这样学习率会对batch size不那么敏感
        SGD(params, learning_rate/batch_size)

        train_loss += nd.mean(loss).asscalar()
        train_acc += accuracy(output,label)

    test_acc = evaluate_accuracy(test_data, net)
    print("Epoch %d, Loss: %f, Train acc: %f, Test acc: %f" %
          (epoch, train_loss/len(train_data), train_acc/len(train_data), test_acc))

# 预测
data, label = mnist_test[0:9]
show_images(data)
print('true labels')
print(get_text_labels(label))

predict_labels = net(data).argmax(axis=1)
print('predict labels')
print(get_text_labels(predict_labels.asnumpy()))

