import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure(figsize=(12, 7))
plt.ion()

xx, yy = np.mgrid[-2:2:.1, -2:2:.1]
grid = np.c_[xx.ravel(), yy.ravel()]
zz = []
for (x, y) in grid:
    z = y * np.math.exp(- x**2 - y**2)
    # z = x**2 + y**2
    zz.append(z)
zz = np.array(zz)
zz = zz.reshape(xx.shape)

# 原始平面
axe1 = fig.add_subplot(121, projection='3d')
axe1.plot_surface(xx, yy, zz, alpha=.5, cmap='rainbow')
sgd_dot = axe1.scatter([0], [0], [0], c='red')
sgdm_dot = axe1.scatter([0], [0], [0], c='green')
adagrad_dot = axe1.scatter([0], [0], [0], c='blue')
rmsprop_dot = axe1.scatter([0], [0], [0], c='cyan')
adam_dot = axe1.scatter([0], [0], [0], c='black')
axe2 = fig.add_subplot(122)
sgd_line, = axe2.plot([0], [0])
sgdm_line, = axe2.plot([0], [0])
adagrad_line, = axe2.plot([0], [0])
rmsprop_line, = axe2.plot([0], [0])
adam_line, = axe2.plot([0], [0])
# line, = axe1.plot(range(1, 10))  # plot时line后面有逗号，scatter时dot后面没有逗号

epochs = 100
lr = 0.1
orig_x = .2
orig_y = .6
# SGD
sgd_x = tf.Variable(tf.constant(orig_x, dtype=tf.float32))
sgd_y = tf.Variable(tf.constant(orig_y, dtype=tf.float32))
opt_sgd_x = []
opt_sgd_y = []
opt_sgd_z = []
# SGDM
sgdm_x = tf.Variable(tf.constant(orig_x, dtype=tf.float32))
sgdm_y = tf.Variable(tf.constant(orig_y, dtype=tf.float32))
opt_sgdm_x = []
opt_sgdm_y = []
opt_sgdm_z = []
m_sgdm_x, m_sgdm_y = 0, 0
beta = 0.9
# Adagrad
adagrad_x = tf.Variable(tf.constant(orig_x, dtype=tf.float32))
adagrad_y = tf.Variable(tf.constant(orig_y, dtype=tf.float32))
opt_adagrad_x = []
opt_adagrad_y = []
opt_adagrad_z = []
v_adagrad_x, v_adagrad_y = 0, 0
# RMSProp
rmsprop_x = tf.Variable(tf.constant(orig_x, dtype=tf.float32))
rmsprop_y = tf.Variable(tf.constant(orig_y, dtype=tf.float32))
opt_rmsprop_x = []
opt_rmsprop_y = []
opt_rmsprop_z = []
v_rmsprop_x, v_rmsprop_y = 0, 0
# Adam
adam_x = tf.Variable(tf.constant(orig_x, dtype=tf.float32))
adam_y = tf.Variable(tf.constant(orig_y, dtype=tf.float32))
opt_adam_x = []
opt_adam_y = []
opt_adam_z = []
m_adam_x, m_adam_y = 0, 0
v_adam_x, v_adam_y = 0, 0
beta2 = 0.999  # beta, beta2 = 0.9, 0.999
global_step = 0

for epoch in range(epochs):
    # sgd
    with tf.GradientTape() as tape:
        sgd_loss = sgd_y * tf.math.exp(- sgd_x**2 - sgd_y**2)  # np.math.exp 时，梯度一直为None
        # loss = x**2 + y**2

        # 用于画图
        opt_sgd_x.append(sgd_x.numpy())
        opt_sgd_y.append(sgd_y.numpy())
        opt_sgd_z.append(sgd_loss.numpy())

    sgd_grads = tape.gradient(sgd_loss, [sgd_x, sgd_y])
    sgd_x.assign_sub(lr * sgd_grads[0])
    sgd_y.assign_sub(lr * sgd_grads[1])
    print("epoch: ", epoch, "sgd_loss:", sgd_loss)

    # sgdm
    with tf.GradientTape() as tape:
        sgdm_loss = sgdm_y * tf.math.exp(- sgdm_x**2 - sgdm_y**2)

        # 用于画图
        opt_sgdm_x.append(sgdm_x.numpy())
        opt_sgdm_y.append(sgdm_y.numpy())
        opt_sgdm_z.append(sgdm_loss.numpy())

    sgdm_grads = tape.gradient(sgdm_loss, [sgdm_x, sgdm_y])
    m_sgdm_x = beta * m_sgdm_x + (1 - beta) * sgdm_grads[0]
    m_sgdm_y = beta * m_sgdm_y + (1 - beta) * sgdm_grads[1]
    sgdm_x.assign_sub(lr * m_sgdm_x)
    sgdm_y.assign_sub(lr * m_sgdm_y)
    print("epoch: ", epoch, "sgdm_loss:", sgdm_loss)

    # adagrad
    with tf.GradientTape() as tape:
        adagrad_loss = adagrad_y * tf.math.exp(- adagrad_x ** 2 - adagrad_y ** 2)

        # 用于画图
        opt_adagrad_x.append(adagrad_x.numpy())
        opt_adagrad_y.append(adagrad_y.numpy())
        opt_adagrad_z.append(adagrad_loss.numpy())

    adagrad_grads = tape.gradient(adagrad_loss, [adagrad_x, adagrad_y])
    v_adagrad_x += tf.square(adagrad_grads[0])
    v_adagrad_y += tf.square(adagrad_grads[1])
    adagrad_x.assign_sub(lr * adagrad_grads[0] / tf.sqrt(v_adagrad_x))
    adagrad_y.assign_sub(lr * adagrad_grads[1] / tf.sqrt(v_adagrad_y))
    print("epoch: ", epoch, "adagrad_loss:", adagrad_loss)

    # RMSProp
    with tf.GradientTape() as tape:
        rmsprop_loss = rmsprop_y * tf.math.exp(- rmsprop_x ** 2 - rmsprop_y ** 2)

        # 用于画图
        opt_rmsprop_x.append(rmsprop_x.numpy())
        opt_rmsprop_y.append(rmsprop_y.numpy())
        opt_rmsprop_z.append(rmsprop_loss.numpy())

    rmsprop_grads = tape.gradient(rmsprop_loss, [rmsprop_x, rmsprop_y])
    v_rmsprop_x = beta * v_rmsprop_x + (1 - beta) * tf.square(rmsprop_grads[0])
    v_rmsprop_y = beta * v_rmsprop_y + (1 - beta) * tf.square(rmsprop_grads[1])
    rmsprop_x.assign_sub(lr * rmsprop_grads[0] / tf.sqrt(v_rmsprop_x))
    rmsprop_y.assign_sub(lr * rmsprop_grads[1] / tf.sqrt(v_rmsprop_y))
    print("epoch: ", epoch, "rmsprop_loss:", rmsprop_loss)

    # Adam
    global_step += 1
    with tf.GradientTape() as tape:
        adam_loss = adam_y * tf.math.exp(- adam_x ** 2 - adam_y ** 2)

        # 用于画图
        opt_adam_x.append(adam_x.numpy())
        opt_adam_y.append(adam_y.numpy())
        opt_adam_z.append(adam_loss.numpy())

    adam_grads = tape.gradient(adam_loss, [adam_x, adam_y])
    m_adam_x = beta * m_adam_x + (1 - beta) * adam_grads[0]
    m_adam_y = beta * m_adam_y + (1 - beta) * adam_grads[1]
    v_adam_x = beta2 * v_adam_x + (1 - beta2) * tf.square(adam_grads[0])
    v_adam_y = beta2 * v_adam_y + (1 - beta2) * tf.square(adam_grads[1])
    m_adam_x_correction = m_adam_x / (1 - tf.pow(beta, int(global_step)))
    m_adam_y_correction = m_adam_y / (1 - tf.pow(beta, int(global_step)))
    v_adam_x_correction = v_adam_x / (1 - tf.pow(beta2, int(global_step)))
    v_adam_y_correction = v_adam_y / (1 - tf.pow(beta2, int(global_step)))
    adam_x.assign_sub(lr * m_adam_x_correction / tf.sqrt(v_adam_x_correction))
    adam_y.assign_sub(lr * m_adam_y_correction / tf.sqrt(v_adam_y_correction))
    print("epoch: ", epoch, "adam_loss:", adam_loss)

    # 开始画图
    sgd_dot.remove()
    sgd_dot = axe1.scatter(opt_sgd_x, opt_sgd_y, opt_sgd_z, c='red', marker='o', label='sgd - o')
    sgdm_dot.remove()
    sgdm_dot = axe1.scatter(opt_sgdm_x, opt_sgdm_y, opt_sgdm_z, c='green', marker='*', label='sgdm - *')
    adagrad_dot.remove()
    adagrad_dot = axe1.scatter(opt_adagrad_x, opt_adagrad_y, opt_adagrad_z, c='blue', marker='+', label='adagrad - +')
    rmsprop_dot.remove()
    rmsprop_dot = axe1.scatter(opt_rmsprop_x, opt_rmsprop_y, opt_rmsprop_z, c='cyan', marker='v', label='rmsprop - v')
    adam_dot.remove()
    adam_dot = axe1.scatter(opt_adam_x, opt_adam_y, opt_adam_z, c='black', marker='s', label='adam - s')
    axe1.legend()
    # 画loss曲线
    sgd_line.remove()
    sgd_line, = axe2.plot(opt_sgd_z, c='red', label='sgd')
    sgdm_line.remove()
    sgdm_line, = axe2.plot(opt_sgdm_z, c='green', label='sgdm')
    adagrad_line.remove()
    adagrad_line, = axe2.plot(opt_adagrad_z, c='blue', label='adagrad')
    rmsprop_line.remove()
    rmsprop_line, = axe2.plot(opt_rmsprop_z, c='cyan', label='rmsprop')
    adam_line.remove()
    adam_line, = axe2.plot(opt_adam_z, c='black', label='adam')
    axe2.set_xlim(0, epochs)
    axe2.set_ylim(-.5, .5)
    axe2.legend()
    plt.pause(.2)


plt.ioff()
plt.show()