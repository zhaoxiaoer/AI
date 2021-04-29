import tensorflow as tf
import matplotlib.pyplot as plt

fig = plt.figure()
plt.ion()

w = tf.Variable(tf.constant(5, dtype=tf.float32))
lr = 0.2
epochs = 10

# loss曲线
# loss = (w + 1)^2
W = tf.range(-6, 6, 0.1)
L = tf.square(W + 1)
axe1 = fig.add_subplot(111)
axe1.plot(W, L)
axe1.scatter(w, tf.square(w + 1), c='red')

for epoch in range(epochs):
    with tf.GradientTape() as tape:
        loss = tf.square(w + 1)
    grads = tape.gradient(loss, w)

    w.assign_sub(lr * grads)
    print("After %d epoch, w is %f, loss is %f" % (epoch, w, loss))

    # 画点
    axe1.scatter(w, tf.square(w + 1), c='red')
    plt.pause(0.5)


plt.ioff()
plt.show()
