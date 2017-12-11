import mxnet as mx

#获取数据集，下载并加载图片和标签至内存
mnist = mx.test_utils.get_mnist()

batch_size = 100

#初始化
#训练集
train_iter = mx.io.NDArrayIter(mnist['train_data'],mnist['train_label'],batch_size,shuffle=True)
#测试集
val_iter = mx.io.NDArrayIter(mnist['train_data'],mnist['train_label'],batch_size)

data = mx.sym.var('data')

################
#定义网络结构

#1st conv layer
#确定数据集，核函数，迭代次数
conv1 = mx.sym.Convolution(data=data,kernel=(5,5),num_filter=20)
#激活函数
tanh1 = mx.sym.Activation(data=conv1,act_type="tanh")
#pooling
pool1 = mx.sym.Pooling(data=tanh1,pool_type="max",kernel=(2,2),stride=(2,2))

#2nd conv layer
conv2 = mx.sym.Convolution(data=pool1,kernel=(5,5),num_filter=50)
tanh2 = mx.sym.Activation(data=conv2,act_type="tanh")
pool2 = mx.sym.Pooling(data=tanh2,pool_type="max",kernel=(2,2),stride=(2,2))

#1st FC layer
flatten = mx.sym.flatten(data=pool2)
fc1 = mx.symbol.FullyConnected(data=flatten,num_hidden=500)
tanh3 = mx.sym.Activation(data=fc1,act_type="tanh")

#2nd FC layer
fc2 = mx.sym.FullyConnected(data=tanh3,num_hidden=10)

#softmax loss
lenet = mx.sym.SoftmaxOutput(data=fc2,name='softmax')

#创建一个训练模型
lenet_model = mx.mod.Module(symbol=lenet,context=mx.gpu(0))
#训练
lenet_model.fit(train_iter,
                eval_data=val_iter,
                optimizer='sgd',optimizer_params={'learning_rate':0.1},
                eval_metric='acc',
                batch_end_callback=mx.callback.Speedometer(batch_size,100), #什么原因
                num_epoch=10)

#预测
test_iter = mx.io.NDArrayIter(mnist['test_data'],None,batch_size)
prob = lenet_model.predict(test_iter)
test_iter = mx.io.NDArrayIter(mnist['test_data'], mnist['test_label'], batch_size)


acc = mx.metric.Accuracy()

lenet_model.score(test_iter,acc)
print(acc)

