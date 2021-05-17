import tensorflow as tf
from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(10, 7))
plt.ion()

x_train = datasets.load_iris().data
y_train = datasets.load_iris().target

np.random.seed(116)
np.random.shuffle(x_train)
np.random.seed(116)
np.random.shuffle(y_train)
np.random.seed(116)

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(3, activation='softmax', kernel_regularizer=tf.keras.regularizers.l2())
])

model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              optimizer=tf.keras.optimizers.SGD(lr=0.1),
              metrics=['sparse_categorical_accuracy'])

history = model.fit(x_train, y_train, batch_size=32, epochs=500, validation_split=.2, validation_freq=1)

model.summary()
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
