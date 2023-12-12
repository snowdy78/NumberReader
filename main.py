import tensorflow as tf
import keras
import random
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Dense, Flatten
from keras.models import Sequential
from keras.datasets import mnist

(X_train, y_train), (X_test, y_test) = mnist.load_data()

#нормализация входных данных
X_train = X_train/255
X_test = X_test/255
# преобразование выходных значений в векторы по категориям
y_train_cat = keras.utils.to_categorical(y_train, 10)
y_test_cat = keras.utils.to_categorical(y_test, 10)


model = Sequential()
# преобразование изображений в входные данные
model.add(Flatten(input_shape=(28, 28, 1)))

# скрытый слой
model.add(Dense(128, activation='relu'))

# выходной слой
model.add(Dense(10, activation='softmax'))


model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
# validation_split - обучающая выборка=80%, выборка валидации=20%
model.fit(X_train, y_train_cat, batch_size=32, epochs=5, validation_split=0.2)

k = random.randint(0, 783)
x = np.expand_dims(X_test[k], axis=0)
res = model.predict(x)
print(res)
plt.text(0.1, 1.1, f"Расспознанная цифра: {np.argmax(res)}")
plt.imshow(X_test[k], cmap='binary')
plt.show()