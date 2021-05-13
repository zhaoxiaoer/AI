import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(10, 7))
plt.ion()

rdm = np.random.RandomState(seed=6725)
x = rdm.rand(32, 2)
y_ = [[4*x1 + 2*x2 + (rdm.rand()*2 - 1)] for (x1, x2) in x]
y_ = np.array(y_)
x = tf.cast(x, tf.float32)

w1 = tf.Variable(tf.random.truncated_normal((2, 20)))
b1 = tf.Variable(tf.constant(0.01, shape=[20]))
w2 = tf.Variable(tf.random.truncated_normal((20, 1)))
b2 = tf.Variable(tf.constant(0.01, shape=[1]))

epochs = 50000
lr = 0.002
train_loss_results = []
lossx = []
lossy = []
lossz = []

for epoch in range(epochs):
    with tf.GradientTape() as tape:
        y = tf.matmul(x, w1) + b1
        # y = tf.nn.relu(y)  # 如果不使用非线性激活函数，神经网络仍然是线性的
        y = tf.matmul(y, w2) + b2
        loss = tf.math.reduce_mean(tf.math.square(y_ - y)) + 0.03*(w1.numpy().sum() + w2.numpy().sum())
        # loss = tf.losses.MSE(y_, y)
    grad = tape.gradient(loss, [w1, b1, w2, b2])
    # print('grad: ', grad)
    w1.assign_sub(lr * grad[0])
    b1.assign_sub(lr * grad[1])
    w2.assign_sub(lr * grad[2])
    b2.assign_sub(lr * grad[3])

    # train_loss_results.append(loss)
    # train_loss_results.append(tf.reduce_mean(loss))

    if epoch % 500 == 0:
        print("-----------epoch %d, loss %f" % (epoch, loss))
        # print(w1.numpy())
        train_loss_results.append(loss)

        fig.clf()

        # 真实数据点
        axe1 = fig.add_subplot(221, projection='3d')
        axe1.scatter(x[:, 0], x[:, 1], y_.ravel())
        # 当前预测平面
        xg, yg = np.mgrid[0:1:0.01, 0:1:0.01]
        grid = np.c_[xg.ravel(), yg.ravel()]
        grid = tf.cast(grid, tf.float32)
        zg = tf.matmul(grid, w1) + b1
        # zg = tf.nn.relu(zg)
        zg = (tf.matmul(zg, w2) + b2).numpy()
        zg = zg.reshape(xg.shape)
        axe1.plot_surface(xg, yg, zg, alpha=0.5)

        # # 损失函数图像
        # axe2 = fig.add_subplot(222, projection='3d')
        # xg, yg = np.mgrid[-20:28:2, -20:24:2]
        # grid = np.c_[xg.ravel(), yg.ravel()]
        # grid = tf.cast(grid, tf.float32)
        # loss = [[tf.math.reduce_mean(tf.math.square(y_ - (tf.matmul(x, [[w11], [w12]]))))] for (w11, w12) in grid]
        # loss = np.array(loss).reshape(xg.shape)
        # axe2.plot_surface(xg, yg, loss, alpha=0.5)
        # # 当前w点
        # lossx.append(w1[0][0])
        # lossy.append(w1[1][0])
        # lossz.append(tf.math.reduce_mean(tf.math.square(y_ - (tf.matmul(x, w1)))))
        # axe2.scatter(lossx, lossy, lossz)

        # 损失函数当前值
        axe3 = fig.add_subplot(223)
        # axe3.set_xlim(0, epochs)
        axe3.set_ylim(0, 1)
        axe3.plot(train_loss_results)
        # w当前值
        axe4 = fig.add_subplot(224)
        axe4.set_xlim(-1, tf.size(w1) + tf.size(w2))
        axe4.set_ylim(-2, 2)
        axe4.bar(range(1, tf.size(w1) + tf.size(w2) + 1), np.concatenate((w1.numpy().ravel(), w2.numpy().ravel())))
        plt.pause(0.2)


plt.ioff()
plt.show()
