import tensorflow as tf
import matplotlib.pyplot as plt
import os

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# fashion = tf.keras.datasets.fashion_mnist
# (x_train, y_train), (x_test, y_test) = fashion.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# 数据增强
# x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)  # 给数据增加一个维度，数据增强时使用的是彩色图片
# image_gen_train = tf.keras.preprocessing.image.ImageDataGenerator(
#     rescale=1. / 1.,
#     rotation_range=45,
#     width_shift_range=.15,
#     height_shift_range=.15,
#     horizontal_flip=True,
#     zoom_range=0.5
# )
# image_gen_train.fit(x_train)

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax', kernel_regularizer=tf.keras.regularizers.l2(0.01))
])

model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              optimizer='adam',
              metrics=['sparse_categorical_accuracy'])

# 断点续训
checkpoint_save_path = "./checkpoint/mnist.ckpt"
if os.path.exists(checkpoint_save_path + '.index'):
    print('------------------load the model---------------------')
    model.load_weights(checkpoint_save_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_save_path,
    save_weights_only=True,
    save_best_only=True
)

# history = model.fit(x_train, y_train, batch_size=32, epochs=50, validation_data=(x_test, y_test), validation_freq=1)
# history = model.fit(image_gen_train.flow(x_train, y_train, batch_size=32), epochs=10,
#                     validation_data=(x_test, y_test), validation_freq=1,
#                     callbacks=[cp_callback])
history = model.fit(x_train, y_train, batch_size=32, epochs=10,
                    validation_data=(x_test, y_test), validation_freq=1,
                    callbacks=[cp_callback])

model.summary()

# plt.imshow(x_train[0], cmap='gray')
fig = plt.figure(figsize=(10, 7))
plt.ion()
# 准确率
axe1 = fig.add_subplot(211)
axe1.plot(history.history['sparse_categorical_accuracy'])
axe1.plot(history.history['val_sparse_categorical_accuracy'])
axe1.set_title('Model accuracy')
axe1.legend(['Train', 'Test'])
# 损失值
axe1 = fig.add_subplot(212)
axe1.plot(history.history['loss'])
axe1.plot(history.history['val_loss'])
axe1.set_title('Model loss')
axe1.legend(['Train', 'Test'])
plt.ioff()
plt.show()
