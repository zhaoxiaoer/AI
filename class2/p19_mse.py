import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure()
plt.ion()

rdm = np.random.RandomState()
x = rdm.rand(32, 3)
y_ = [[13*x1 + 4*x2 + 4*x3 + (rdm.rand() - 0.5)] for (x1, x2, x3) in x]
x = tf.cast(x, tf.float32)

w1 = tf.Variable(tf.random.truncated_normal((3, 1)))

epochs = 15000
lr = 0.002
train_loss_results = []

for epoch in range(epochs):
    with tf.GradientTape() as tape:
        y = tf.matmul(x, w1)
        loss = tf.math.reduce_mean(tf.math.square(y_ - y))
    grad = tape.gradient(loss, w1)
    w1.assign_sub(lr * grad)

    train_loss_results.append(loss)

    if epoch % 500 == 0:
        print("epoch %d" % epoch)
        print(w1.numpy())

        fig.clf()
        axe1 = fig.add_subplot(211)
        axe1.set_xlim(0, 15000)
        axe1.set_ylim(0, 1)
        axe1.plot(train_loss_results)
        axe2 = fig.add_subplot(212)
        axe2.set_xlim(0, 3)
        axe2.set_ylim(-25, 25)
        axe2.bar(range(w1.numpy().size), w1.numpy().ravel())
        plt.pause(0.2)


plt.ioff()
plt.show()
