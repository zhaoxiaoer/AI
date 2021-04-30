# -*- coding: UTF-8 -*-
# 利用鸢尾花数据集，实现前向传播、反向传播，可视化loss曲线

import tensorflow as tf
from sklearn import datasets
from matplotlib import pyplot as plt
import numpy as np

fig = plt.figure()
plt.ion()

# 导入数据，分别为输入特征和标签
x_data = datasets.load_iris().data
y_data = datasets.load_iris().target

# 随机打算数据
np.random.seed(116)
np.random.shuffle(x_data)
np.random.seed(116)
np.random.shuffle(y_data)
tf.random.set_seed(116)

# 将数据集分割为训练集和测试集
x_train = x_data[:-30]
y_train = y_data[:-30]
x_test = x_data[-30:]
y_test = y_data[-30:]

# 转换数据类型
x_train = tf.cast(x_train, tf.float32)
x_test = tf.cast(x_test, tf.float32)

# 配成（特征，标签）对，每次读入一小撮
train_db = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(32)
test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

# 定义神经网络参数
w1 = tf.Variable(tf.random.truncated_normal([4, 3], stddev=0.1, seed=1))
b1 = tf.Variable(tf.random.truncated_normal([3], stddev=0.1, seed=1))

lr = 0.1
train_loss_results = []
test_acc = []
epochs = 500
loss_all = 0

for epoch in range(epochs):
    for step, (x_train_b, y_train_b) in enumerate(train_db):
        with tf.GradientTape() as tape:
            y = tf.matmul(x_train_b, w1) + b1  # 神经网络乘加运算
            y = tf.nn.softmax(y)  # 使输出y符合概率分布（此操作后与独热码同量级，可相减求loss）
            y_ = tf.one_hot(y_train_b, depth=3)  # 将标签值转换为独热码格式，方便计算loss和accuracy
            loss = tf.reduce_mean(tf.square(y_ - y))  # 采用均方误差损失函数mse
            loss_all += loss.numpy()  # 将每个step计算出的loss累加，为后续求loss平均值提供数据，这样计算的loss更准确
        # 计算loss对各个参数的梯度
        grads = tape.gradient(loss, [w1, b1])

        # 实现梯度更新
        w1.assign_sub(lr * grads[0])
        b1.assign_sub(lr * grads[1])

    # 每个epoch，打印loss信息
    print("Epoch {}, loss: {}".format(epoch, loss_all/4))
    train_loss_results.append(loss_all / 4)  # 将4个step的loss求平均记录在此变量中
    loss_all = 0  # loss_all归零，为记录下一个epoch的loss做准备

    # 测试部分
    # total_correct为预测对的样本个数，total_number为测试的总样本数，将这两个变量都初始化为0
    total_correct, total_number = 0, 0
    for x_test_b, y_test_b in test_db:
        # 使用更新后的参数进行预测
        y = tf.matmul(x_test_b, w1) + b1
        y = tf.nn.softmax(y)
        pred = tf.argmax(y, axis=1)  # 返回y中最大值的索引，即预测的分类
        # 将pred转换为y_test的数据类型
        pred = tf.cast(pred, dtype=y_test_b.dtype)
        # 若分类正确，则correct=1，否则为0，将bool型的结果转换为int型
        correct = tf.cast(tf.equal(pred, y_test_b), tf.int32)
        # 将每个batch的correct数加起来
        correct = tf.reduce_sum(correct)
        # 将所有batch中的correct数加起来
        total_correct += int(correct)
        total_number += x_test_b.shape[0]
    # 总的准确率等于total_correct/total_number
    acc = total_correct / total_number
    test_acc.append(acc)
    print("Test_acc: ", acc)
    print("--------------")

    fig.clf()
    axe1 = fig.add_subplot(411)
    axe1.plot(train_loss_results)
    axe1.set_xlim(0, 500)
    # axw1.set_ylim(0, 1)
    axe2 = fig.add_subplot(412)
    axe2.plot(test_acc)
    axe2.set_xlim(0, 500)
    axe2.set_ylim(0, 1.2)
    axe3 = fig.add_subplot(413)
    axe3.bar(range(w1.numpy().size), w1.numpy().reshape(-1, 1).squeeze())
    axe4 = fig.add_subplot(414)
    axe4.bar(range(b1.numpy().size), b1.numpy())

    plt.pause(0.2)


plt.ioff()
plt.show()