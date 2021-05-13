import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

fig = plt.figure(figsize=(10, 7))
plt.ion()

df = pd.read_csv('./dot.csv')
x_data = np.array(df[['x1', 'x2']])
y_data = np.array(df['y_c'])

x_train = np.vstack(x_data).reshape(-1, 2)
y_train = np.vstack(y_data).reshape(-1, 1)

Y_c = [['red' if y else 'blue'] for y in y_train]

x_train = tf.cast(x_train, tf.float32)
y_train = tf.cast(y_train, tf.float32)

train_db = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(32)

w1 = tf.Variable(tf.random.normal([2, 11]), dtype=tf.float32)
b1 = tf.Variable(tf.constant(0.01, shape=[11]))
w2 = tf.Variable(tf.random.normal([11, 1]), dtype=tf.float32)
b2 = tf.Variable(tf.constant(0.01, shape=[1]))

lr = 0.01
epochs= 800

# 原始数据点画图
# 三维
axe1 = fig.add_subplot(221, projection='3d')
x1 = x_data[:, 0]
x2 = x_data[:, 1]
axe1.scatter(x1, x2, y_data)
# 二维
axe2 = fig.add_subplot(222)
axe2.scatter(x1, x2, color=np.ravel(Y_c))
# 生成网格坐标点，显示预测值
xx, yy = np.mgrid[-3:3:.1, -3:3:.1]
grid = np.c_[xx.ravel(), yy.ravel()]
grid = tf.cast(grid, tf.float32)
# 显示w
axe3 = fig.add_subplot(212)
axe3.set_xlim(-1, tf.size(w1) + tf.size(w2))

for epoch in range(epochs):
    for step, (x_train, y_train) in enumerate(train_db):
        with tf.GradientTape() as tape:
            h1 = tf.matmul(x_train, w1) + b1
            h1 = tf.nn.relu(h1)
            y = tf.matmul(h1, w2) + b2

            # loss = tf.reduce_mean(tf.square(y_train - y))
            # L2 正则化
            loss_mse = tf.reduce_mean(tf.square(y_train - y))
            # 添加l2正则化
            loss_regularization = []
            # tf.nn.l2_loss(w) = sum(w ** 2) / 2
            loss_regularization.append(tf.nn.l2_loss(w1))
            loss_regularization.append(tf.nn.l2_loss(w2))
            # 求和
            loss_regularization = tf.reduce_sum(loss_regularization)
            loss = loss_mse + 0.03 * loss_regularization  # REGULARIZER = 0.03

        variables = [w1, b1, w2, b2]
        grad = tape.gradient(loss, variables)

        w1.assign_sub(lr * grad[0])
        b1.assign_sub(lr * grad[1])
        w2.assign_sub(lr * grad[2])
        b2.assign_sub(lr * grad[3])

    if epoch % 20 == 0:
        print('epoch:', epoch, 'loss:', float(loss))

        axe2.clear()
        axe2.scatter(x1, x2, color=np.ravel(Y_c))
        # 等高线图
        probs = []
        for x_test in grid:
            h1 = tf.matmul([x_test], w1) + b1
            h1 = tf.nn.relu(h1)
            y = tf.matmul(h1, w2) + b2
            probs.append(y)
        probs = np.array(probs).reshape(xx.shape)
        C = axe2.contour(xx, yy, probs, levels=[.5])
        # C = axe2.contour(xx, yy, probs)
        axe2.clabel(C, inline=True, fontsize=10)
        axe3.clear()
        axe3.bar(range(1, w1.numpy().size + w2.numpy().size + 1), np.concatenate((w1.numpy().ravel(), w2.numpy().ravel())))
        axe3.set_ylim(-1, 1)
        # 必须有pause，否则无法显示
        plt.pause(.2)


plt.ioff()
plt.show()
