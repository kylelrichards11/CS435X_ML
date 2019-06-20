# Machine Learning Homework 3
# mapanicci
# krichards

import numpy as np
import skimage as sk
import matplotlib.pyplot as plt
import pandas as pd

########################################################################################################################
# Gradient Descent Training

def makePredictions(X, w):
    return np.dot(X.T, w)

def loadData(which):
    digits = np.load("small_mnist_{}_images.npy".format(which))
    digits = digits.reshape(-1, 28, 28)
    labels = np.load("small_mnist_{}_labels.npy".format(which))
    return digits, labels

def showImage(img):
    plt.imshow(img, cmap='gray')
    plt.show()

def printResults(y, yhat):
    print("PC = ", np.round(percentCorrect(y, yhat) * 100, 3), " CE = ", np.round(crossEntropy(y, yhat), 4))

def makeX(data):
    shape = data.shape
    X = data.reshape(shape[0], shape[1]*shape[2])
    X = np.transpose(X)
    b = np.ones(shape[0])
    return np.vstack((X, b))

def zFunction(yhat):
    yhat = np.exp(yhat)
    yhatSum = np.sum(yhat, axis=1).reshape(-1, 1)
    return yhat/yhatSum

def getGradient(X, w, y):
    yhat = np.dot(X.T, w)
    yhat = zFunction(yhat)
    return (1/X.shape[1])*np.dot(X, yhat - y.T)

def gradientDescent(X, y, w):
    learningRate = 0.1
    gradient = getGradient(X, w, y)
    w = w - learningRate*gradient
    return w

def crossEntropy(y, yhat):
    return -np.sum(y.T * np.log(yhat)) / len(y.T)

def percentCorrect(y, yhat):
    return sum(np.argmax(yhat, axis=1) == np.argmax(y, axis=0)) / yhat.shape[0]

def stochGradDescent(trainingDigits, trainingLabels):
    std_dev = 0.01
    epochs = 100
    batchSize = 50

    X = makeX(trainingDigits)
    X_wLabels = np.vstack((X, trainingLabels.T))
    np.random.shuffle(X_wLabels.T)
    X = X_wLabels[0:-10,:]
    y = X_wLabels[-10:,:]
    splitX = np.hsplit(X, batchSize)
    splitY = np.hsplit(y, batchSize)

    w = std_dev * np.random.randn(X.shape[0], y.shape[0])

    for i in range(epochs):
        for x_block, y_block in zip(splitX, splitY):
            w = gradientDescent(x_block, y_block, w)

        testModel(trainingDigits, trainingLabels, w)
    return w

def testModel(data, labels, w):
    testX = makeX(data)
    yhat = makePredictions(testX, w)
    printResults(labels.T, zFunction(yhat))

########################################################################################################################
# Augmentation

def translation(images): # Help from https://stackoverflow.com/questions/47961447/shift-image-in-scikit-image-python
    translated = []
    for img in images:
        scale = np.random.uniform(-2, 2, size=2)
        transform = sk.transform.AffineTransform(translation=scale)
        shifted = sk.transform.warp(img, transform, preserve_range=True)
        translated.append(shifted.astype(img.dtype))
    return np.array(translated)

def rotation(images):
    rotated = []
    for img in images:
        rotated.append(sk.transform.rotate(img, np.random.uniform(-30, 30)))
    return np.array(rotated)

def scale(images):
    scaled = []
    scale = np.random.uniform(0.5, 1.15)
    for img in images:
        newImg = sk.transform.rescale(img, scale, multichannel=False, anti_aliasing=True, mode='constant')
        if scale > 1:
            center = int(newImg.shape[0] / 2)
            scaled.append(newImg[center-14:center+14, center-14:center+14])
        else:
            base = np.zeros(img.shape)
            center = int(base.shape[0] / 2)
            dist_add = int(newImg.shape[0]/2)
            dist_sub = int((newImg.shape[0]/2) + 0.5)
            base[center-dist_sub:center+dist_add, center-dist_sub:center+dist_add] = newImg
            scaled.append(base)
    return np.array(scaled)


def noise(images):
    noisey = []
    for img in images:
        noise_im = np.copy(img)
        non_zero_size = noise_im[noise_im!=0].shape
        noise = 0.2*np.sqrt(np.random.randn(non_zero_size[0])**2)
        noise_im[noise_im != 0] = noise_im[noise_im != 0] + noise
        noise_im = noise_im / np.max(noise_im)
        noisey.append(noise_im)
    return np.array(noisey)

def augmentData(data, labels):
    rotated = rotation(trainingDigits)
    scaled = scale(trainingDigits)
    translated = translation(trainingDigits)
    noisey = noise(trainingDigits)

    showAugmentations = False
    if showAugmentations:
        showImage(data[15, :, :])
        showImage(rotated[15, :, :])
        showImage(data[11, :, :])
        showImage(scaled[11, :, :])
        showImage(data[109, :, :])
        showImage(translated[109, :, :])
        showImage(data[111, :, :])
        showImage(noisey[111, :, :])

    augmentedData = np.vstack((data, rotated, scaled, translated, noisey))
    augmentedLabels = np.vstack((labels, labels, labels, labels, labels))

    return augmentedData, augmentedLabels

########################################################################################################################
# Main

if __name__ == "__main__":
    np.random.seed(2)

    trainingDigits, trainingLabels = loadData("train")
    testingDigits, testingLabels = loadData("test")

    # Regular Gradient Descent
    print("\nOriginal Training Data")
    w = stochGradDescent(trainingDigits, trainingLabels)
    print("Training Accuracy")
    testModel(trainingDigits, trainingLabels, w)
    print("Testing Accuracy")
    testModel(testingDigits, testingLabels, w)

    # Augmentation
    print("\nWith Augmentation")
    augmentedData, augmentedLabels = augmentData(trainingDigits, trainingLabels)
    w = stochGradDescent(augmentedData, augmentedLabels)

    print("Training Accuracy")
    testModel(augmentedData, augmentedLabels, w)
    print("Testing Accuracy")
    testModel(testingDigits, testingLabels, w)