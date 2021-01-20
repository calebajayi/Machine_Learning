# import libraries
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # ignore error for gpu with tensorflow if you dont have a gpu
import numpy as np
import matplotlib.pyplot as plt
import keras

fashion_mnist = keras.datasets.fashion_mnist  # load the dataset
# split the data into testing and training
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
# print(train_images.shape)
# print(train_images[0, 23, 23])
# print(train_images[:10])

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
# plt.figure2()
# plt.imshow(train_images[1])
# plt.colorbar()
# plt.grid(False)
# plt.show()

train_images = train_images / 255.0
test_images = test_images / 255.0

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),  # Hidden layer
    keras.layers.Dense(10, activation='softmax')  # Output layer
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=10)

test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=1)
print('test accuracy:', test_acc)  # print test accuracy

predictions = model.predict(test_images)
# print(predictions[0])
# print(np.argmax(predictions[0]))  # simply returns the index of the maximum value from a numpy array.
# print(test_labels[0])  # we can check if this is correct by looking at the value of the co-responding test label.
# print(class_names[np.argmax(predictions[2])])  # what are we predicting the image is?
# plt.figure()
# plt.imshow(test_images[2])  # what the image actually is
# plt.colorbar()
# plt.grid(False)
# plt.show()

# Function to predict the output of the model depending on user number input
COLOR = 'white'
plt.rcParams['text.color'] = COLOR
plt.rcParams['axes.labelcolor'] = COLOR


def predict(model, image, correct_label):
    global class_names
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    prediction = model.predict(np.array([image]))
    predicted_class = class_names[np.argmax(prediction)]

    show_image(image, class_names[correct_label], predicted_class)


def show_image(img, label, guess):
    plt.figure()
    plt.imshow(img, cmap=plt.cm.binary)
    print("Expected: ", label)
    print("Guess: ", guess)
    plt.colorbar()
    plt.grid(False)
    plt.show()


def get_number():
    while True:
        num = input("Pick a number: ")
        if num.isdigit():
            num = int(num)
            if 0 <= num <= 1000:
                return int(num)
        else:
            print("Try again...")


num = get_number()
image = test_images[num]
label = test_labels[num]
predict(model, image, label)