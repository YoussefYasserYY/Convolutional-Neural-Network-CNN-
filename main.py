from tensorflow import keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense

# loading data
(X_train, y_train), (X_test, y_test) = mnist.load_data()
# reshaping data
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], X_train.shape[2], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], X_test.shape[2], 1))
# checking the shape after reshaping
print(X_train.shape)
print(X_test.shape)
# normalizing the pixel values
X_train = X_train.astype("float32") / 255
X_test = X_test.astype("float32") / 255

# defining model
model = Sequential()
# adding convolution layer
model.add(Conv2D(70, kernel_size=(3, 3), activation='relu', strides=(2, 2), input_shape=(28, 28, 1), padding='same'))
model.add(MaxPool2D(pool_size=(2, 2), padding='same'))

model.add(Conv2D(70, kernel_size=(3, 3), activation='relu', strides=(2, 2), padding='same'))
model.add(MaxPool2D(pool_size=(2, 2), padding='same'))

# adding fully connected layer
model.add(Flatten())
model.add(Dropout(rate=0.15))
model.add(Dense(80, activation='relu'))
model.add(Dense(80, activation='relu'))
model.add(Dense(10, activation='softmax'))
# compiling the model
optimizer = keras.optimizers.Adagrad(learning_rate=0.05)
model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
# fitting the model
model_hist = model.fit(X_train, y_train, epochs=15, shuffle="true", batch_size=32, validation_data=(X_test, y_test))
acc = model_hist.history['accuracy']
val_acc = model_hist.history['val_accuracy']
summary = model.evaluate(X_test, y_test, batch_size=32)
print('Final accuracy of the model: %.3f' % (summary[1] * 100.0), end="%\n")
print("the accuracy in the first 5 epoch train: ", end=" ")
print('%.3f' % (acc[0] * 100.0), end="% -> ")
print('%.3f' % (acc[1] * 100.0), end="% -> ")
print('%.3f' % (acc[2] * 100.0), end="% -> ")
print('%.3f' % (acc[3] * 100.0), end="% -> ")
print('%.3f' % (acc[4] * 100.0), end="%\n")
print("the accuracy in the first 5 epoch test: ", end=" ")
print('%.3f' % (val_acc[0] * 100.0), end="% -> ")
print('%.3f' % (val_acc[1] * 100.0), end="% -> ")
print('%.3f' % (val_acc[2] * 100.0), end="% -> ")
print('%.3f' % (val_acc[3] * 100.0), end="% -> ")
print('%.3f' % (val_acc[4] * 100.0), end="%\n")
print('\n')
