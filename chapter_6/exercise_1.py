import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.utils.np_utils import to_categorical
from keras.layers import MaxPooling2D, AveragePooling2D, GlobalMaxPooling2D, Dense, Conv2D, Flatten, Activation
from keras.models import Sequential

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train_flat = X_train.reshape((60000, 784))
X_test_flat = X_test.reshape(-1, 28*28)

X_train_sc = X_train_flat.astype('float32') / 255.0
X_test_sc = X_test_flat.astype('float32') / 255.0

y_train_cat = to_categorical(y_train)
y_test_cat = to_categorical(y_test)

X_train_t = X_train_sc.reshape(-1, 28, 28, 1)
X_test_t = X_test_sc.reshape(-1, 28, 28, 1)

model = Sequential()

model.add(Conv2D(32, (3, 3), kernel_initializer='normal', input_shape=(28, 28, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

#model.summary()

h = model.fit(X_train_t, y_train_cat, batch_size=128,
              epochs=5, verbose=1, validation_split=0.3)

plt.plot(h.history['acc'])
plt.plot(h.history['val_acc'])
plt.legend(['Training', 'Validation'])
plt.title('Accuracy')
plt.xlabel('Epochs')
plt.show()

train_acc = model.evaluate(X_train_t, y_train_cat, verbose=0)[1]
test_acc = model.evaluate(X_test_t, y_test_cat, verbose=0)[1]

print("Train accuracy: {:0.4f}, Test accuracy: {:0.4f}".format(train_acc, test_acc))

# does performance improve?
# No

# how many parameters does this new model have? More or less than the previous model? Why?
# Less as we have added a filter layer that decresed the parameters

# how long did this second model take to train? Longer or shorter than the previous model? Why?
# Dont' know

# did it perform better or worse than the previous model?
# Worse
