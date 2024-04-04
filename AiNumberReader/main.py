import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageDraw, ImageOps
from skimage.filters import threshold_otsu, threshold_mean
import cv2

data = pd.read_csv('./train.csv')

data = np.array(data)
m, n = data.shape
np.random.shuffle(data)

data_dev = data[0:1000].T
Y_dev = data_dev[0]
X_dev = data_dev[1:n]
X_dev = X_dev / 255

data_train = data[1000:m].T
Y_train = data_train[0]
X_train = data_train[1:n]

def binarize_images(images):
    threshold_value = 128
    binarized_images = np.where(images > threshold_value, 255, 0)
    return binarized_images

# Apply thresholding to training data
X_train_binarized = binarize_images(X_train)
X_train = X_train / 255
X_train_binarized = X_train_binarized / 255
X_train = X_train_binarized


def init_params():
    W1 = np.random.rand(10, 784) - 0.5
    b1 = np.random.rand(10, 1) - 0.5
    W2 = np.random.rand(10, 10) - 0.5
    b2 = np.random.rand(10, 1) - 0.5
    return W1, b1, W2, b2


def ReLU(z):
    return np.maximum(z, 0)


def softmax(z):
    z -= np.max(z, axis=0)  # Subtract the maximum value for numerical stability
    return np.exp(z) / np.sum(np.exp(z), axis=0)


def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2


def deriv_relu(z):
    return z > 0


def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y


def back_prop(Z1, A1, Z2, A2, W1, W2, X, Y):
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2)
    dZ1 = W2.T.dot(dZ2) * deriv_relu(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1)
    return dW1, db1, dW2, db2


def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1
    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2
    return W1, b1, W2, b2


def get_predictions(A2):
    return np.argmax(A2, 0)


def get_accuracy(predictions, Y):
    return np.sum(predictions == Y) / Y.size


def gradient_descent(X, Y, iterations, alpha):
    W1, b1, W2, b2 = init_params()
    for i in range(iterations):

        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = back_prop(Z1, A1, Z2, A2, W1, W2, X, Y)
        W1, b1, W2, b1 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        if (i % 50 == 0):
            print("Iteration: ", i)
            print("Accuracy: ", get_accuracy(get_predictions(A2), Y))
            print("")
    return W1, b1, W2, b2

W1, b1, W2, b2 = gradient_descent(X_train, Y_train, 2500, 0.2)

_, m_train = X_train.shape


def make_predictions(X, W1, b1, W2, b2):
    _, _, _, A2 = forward_prop(W1, b1, W2, b2, X)
    predictions = get_predictions(A2)
    return predictions


def test_prediction(index, W1, b1, W2, b2):
    current_image = X_train[:, index, None]
    prediction = make_predictions(X_train[:, index, None], W1, b1, W2, b2)
    label = Y_train[index]
    print("Prediction: ", prediction)
    print("Label: ", label)

    current_image = current_image.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()

def draw_and_predict():
    def motion(event):
        x, y = event.x, event.y
        r = 12
        canvas.create_oval(x - r, y - r, x + r, y + r, fill="black")
        draw.ellipse([x - r, y - r, x + r, y + r], fill="black")


    def predict():
        # Convert the drawn image to a format compatible with the model
        image_resized = image.resize((28, 28))

        # Convert the image to grayscale
        image_resized_gray = image_resized.convert('L')

        threshold_value = 128
        image_array = np.array(image_resized_gray)
        image_thresholded_array = np.where(image_array > threshold_value, 255, 0)

        # Convert the thresholded array back to an image
        image_thresholded = Image.fromarray(image_thresholded_array.astype('uint8'))

        # Invert the colors
        image_inverted = ImageOps.invert(image_thresholded)

        # Plot the inverted image for debugging
        # plt.gray()
        # plt.imshow(image_inverted, interpolation='nearest')
        # plt.title("Inverted Image")
        # plt.show()

        # Reshape the image to match the input shape
        data = np.array(image_inverted) / 255.0
        data = data.reshape((1, 784)).T

        # Perform prediction using your model
        prediction = make_predictions(data, W1, b1, W2, b2)
        messagebox.showinfo("Prediction", f"The drawn number is: {prediction}")

    popup = tk.Toplevel()
    popup.title("Draw a Number")
    canvas = tk.Canvas(popup, width=280, height=280, bg="white")
    canvas.pack()


    image = Image.new("L", (280, 280), "white")
    draw = ImageDraw.Draw(image)


    canvas.bind("<B1-Motion>", motion)


    predict_button = tk.Button(popup, text="Predict", command=predict)
    predict_button.pack()

root = tk.Tk()
root.title("Number Recognition")

draw_button = tk.Button(root, text="Draw Number", command=draw_and_predict)
draw_button.pack()

root.mainloop()


