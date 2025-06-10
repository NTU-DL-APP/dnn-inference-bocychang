# train_model.py
import tensorflow as tf
import numpy as np
import json
import os

# 1. 載入資料
fashion_mnist = tf.keras.datasets.fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# 2. 建立模型
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 3. 訓練模型
model.fit(x_train, y_train, epochs=20, validation_data=(x_test, y_test))

# 4. 儲存模型
if not os.path.exists('model'):
    os.makedirs('model')
model.save('model/fashion_mnist.h5')

# 5. 儲存架構
with open('model/fashion_mnist.json', 'w') as f:
    json.dump(json.loads(model.to_json()), f)

# 6. 儲存權重
weights = {}
for layer in model.layers:
    for i, w in enumerate(layer.get_weights()):
        weights[f"{layer.name}_weight_{i}"] = w
np.savez('model/fashion_mnist.npz', **weights)