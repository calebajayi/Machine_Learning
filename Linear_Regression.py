# import libraries
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # ignore error for gpu with tensorflow if you dont have a gpu
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import wget

print(tf.__version__)

# get the data
url = 'https://raw.githubusercontent.com/lazyprogrammer/machine_learning_examples/master/tf2.0/moore.csv'
moore = wget.download(url)

# load the data
data = pd.read_csv('moore.csv', header=None).values  # ignore the header columns
X = data[:, 0].reshape(-1, 1)  # make it a 2-d array of size N * D where D = 1
Y = data[:, 1]
plt.scatter(X, Y)
plt.show()

# Since we want a linear model we take the log
Y = np.log(Y)
plt.scatter(X, Y)
plt.show()

# center the X data so the values are not too large
X = X - X.mean()

# Create the Tensorflow Model
model = tf.keras.models.Sequential([tf.keras.layers.Input(shape=(1,)), tf.keras.layers.Dense(1)])
model.compile(optimizer=tf.keras.optimizers.SGD(0.001, 0.9), loss='mse')


# model.compile(optimizer='adam', loss='mse')

# learning rate scheduler
def schedule(epoch, lr):
    if epoch >= 50:
        return 0.0001
    return 0.001


scheduler = tf.keras.callbacks.LearningRateScheduler(schedule)

# train the model
r = model.fit(X, Y, epochs=200, callbacks=[scheduler])

# plot the loss
plt.plot(r.history['loss'], label='loss')
plt.show()

# get the slope of the line
# the slope of the line is related to the doubling rate of transistor count
print(model.layers)  # there is only 1 layer, input layer doesnt count
print(model.layers[0].get_weights())

# the scope of the line is:
a = model.layers[0].get_weights()[0][0, 0]
print("Time to double:", np.log(2) / a)
