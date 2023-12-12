
import tensorflow as tf
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from keras.models import Sequential
import scipy

ImageDataGenerator = tf.keras.preprocessing.image.ImageDataGenerator
TRAINING_DIR = "Cyrillic"
train_datagen = ImageDataGenerator(rescale=1.0 / 255.)
train_generator = train_datagen.flow_from_directory(TRAINING_DIR,
                              batch_size=40,
                              class_mode='binary',
                              target_size=(278,278))
VALIDATION_DIR = "Cyrillic"
validation_datagen = ImageDataGenerator(rescale=1.0 / 255.)
validation_generator = validation_datagen.flow_from_directory(VALIDATION_DIR,
                                      batch_size=40,
                                      class_mode='binary',
                                      target_size=(278,278))
model = Sequential([
    Conv2D(16, (3, 3), activation='relu',
                input_shape=(278,278, 3)),
    MaxPooling2D(2, 2),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(33, activation='softmax')
])
model.compile(optimizer='adam',
loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()
history = model.fit(train_generator, epochs=2, verbose=1, validation_data=validation_generator)
model.save("model.h5")