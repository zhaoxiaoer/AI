import tensorflow as tf
import matplotlib.pyplot as plt

fig = plt.figure()
plt.ion()

w1 = tf.Variable(tf.constant(-5, dtype=tf.float32))
w2 = tf.Variable(tf.constant(5, dtype=tf.float32))
lr = 0.2
epochs = 10

# loss曲线
# loss = w1^2 + w2^2
W1 = tf.range(-6, 6, 0.1)
W2 = tf.range(-6, 6, 0.1)
WW1, WW2 = tf.meshgrid(W1, W2)
L = tf.square(WW1) + tf.square(WW2)
axe1 = fig.add_subplot(111, projection='3d')
axe1.plot_surface(WW1, WW2, L, alpha=0.7)
axe1.scatter(w1, w2, tf.square(w1) + tf.square(w2), c='r', s=20, marker='o')

for epoch in range(epochs):
    with tf.GradientTape() as tape:
        loss = tf.square(w1) + tf.square(w2)
    grads = tape.gradient(loss, [w1, w2])

    w1.assign_sub(lr * grads[0])
    w2.assign_sub(lr * grads[1])
    print("After %d epoch, w1 is %f, w2 is %f, loss is %f" % (epoch, w1, w2, loss))

    # 画点
    axe1.scatter(w1, w2, tf.square(w1) + tf.square(w2), c='r', s=10, marker='o')
    plt.pause(0.5)


plt.ioff()
plt.show()
