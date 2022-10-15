#import Statements
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Flatten


# Load Data
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

# Pre Process the images
X_train = X_train.astype(np.float32)/255
X_test = X_test.astype(np.float32)/255

# converting y to one hot vectors
y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)

# # Build sequential Model
model = Sequential(
    [
        # Flatten and specify input size
        Flatten(input_shape=(28, 28)),
        # Add layers
        Dense(128, activation='relu', name='reluLayer1', input_shape=(28, 28)),
        Dense(128, activation='relu', name='reluLayer2'),
        # Output layer
        Dense(10, activation='softmax', name='softmaxLayer')

    ], name="handwritten_digit_recoginition_model"
)


# Compile the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.categorical_crossentropy, metrics=['accuracy'])

# fit the training data
model.fit(X_train, y_train, epochs=52)
print("The model has successfully trained")

# Test the model on testing set
score = model.evaluate(X_test, y_test)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

model.save('handwritten.model')
print("Saving the model as handwritten.model")
