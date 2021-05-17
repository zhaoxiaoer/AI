import tensorflow as tf
import matplotlib.pyplot as plt

# mnist = tf.keras.datasets.mnist
# (x_train, y_train), (x_test, y_test) = mnist.load_data()
fashion = tf.keras.datasets.fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax', kernel_regularizer=tf.keras.regularizers.l2(0.01))
])

model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              optimizer='adam',
              metrics=['sparse_categorical_accuracy'])

history = model.fit(x_train, y_train, batch_size=32, epochs=50, validation_data=(x_test, y_test), validation_freq=1)

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
