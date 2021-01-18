import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import sklearn
from sklearn.datasets import load_breast_cancer

print(sklearn.__version__)
print(tf.__version__)

# load in the data


# save it in a variable
data = load_breast_cancer()
# check the type of data
print(type(data))
# note it is a bunch object
# this basically treats the keys as attributes
print(data.keys())
# 'data' (the attribute) means the input data
print(data.data.shape)
# 'targets'
print(data.target)
print(data.target_names)
# there are 569 input targets
print(data.target.shape)

# normally all the imports go on top
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.33)
N, D = X_train.shape
# Scale the data
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(D,)),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# train the model
r = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100)


# evaluate the model - evaluat() returns  loss and accuracy
print("Train score:", model.evaluate(X_train, y_train))
print("Test score:", model.evaluate(X_test, y_test))

# plot whats returned by the model fit
import matplotlib.pyplot as plt

plt.plot(r.history['loss'], label='loss')
plt.plot(r.history['val_loss'], label='val_loss')
plt.legend()

# plot the accuracy too
plt.plot(r.history['accuracy'], label='acc')
plt.plot(r.history['val_accuracy'], label='val_acc')
plt.legend()
