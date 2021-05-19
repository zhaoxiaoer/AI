import tensorflow as tf
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

model_save_path = '../class3/checkpoint/mnist.ckpt'

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.load_weights(model_save_path)

preNum = int(input("input the number of test pictures:"))
for e in range(preNum):
    image_path = input("the path of test picture:")
    img = Image.open(image_path)
    plt.imshow(img)
    plt.show()
    img = img.resize((28, 28), Image.ANTIALIAS)
    img_arr = np.array(img.convert('L'))

    # 数据预处理之前，测试效果很差，预处理之后，测试效果比较好
    # img_arr = 255 - img_arr  # 颜色取反，在白底黑字图片上需要进行转换
    for i in range(28):
        for j in range(28):
            if img_arr[i][j] < 200:
                img_arr[i][j] = 255
            else:
                img_arr[i][j] = 0

    img_arr = img_arr / 255.0
    print("img_arr: ", img_arr.shape)
    x_predict = img_arr[tf.newaxis, ...]
    print("x_predict: ", x_predict.shape)
    result = model.predict(x_predict)

    pred = tf.argmax(result, axis=1)
    print('\n')
    tf.print(pred)
