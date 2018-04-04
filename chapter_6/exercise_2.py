import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist, cifar10
from keras.utils.np_utils import to_categorical
from keras.layers import MaxPooling2D, AveragePooling2D, GlobalMaxPooling2D, Dense, Conv2D, Flatten, Activation
from keras.models import Sequential

#plt.imshow(first_img, cmap='gray')
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

#plt.imshow(X_train[0])

# X_train.shape (10000, 32, 32, 3)
# y_train.shape (50000, 1)

X_train_flat = X_train.reshape((-1, 32*32*3)) 
X_test_flat = X_test.reshape(-1, 32*32*3)

X_train_sc = X_train_flat.astype('float32') / 255.0
X_test_sc = X_test_flat.astype('float32') / 255.0

y_train_cat = to_categorical(y_train)
y_test_cat = to_categorical(y_test)

X_train_t = X_train_sc.reshape(-1, 32, 32, 3)
X_test_t = X_test_sc.reshape(-1, 32, 32, 3)

model = Sequential()

model.add(Conv2D(32, (3, 3), kernel_initializer='normal', input_shape=(32, 32, 3)))
model.add(Conv2D(32, (3, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(16, (2, 2)))
model.add(Conv2D(16, (2, 2)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, (3, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model.summary()

h = model.fit(X_train_t, y_train_cat, batch_size=128,
              epochs=5, verbose=1, validation_split=0.3)

# Too slow
