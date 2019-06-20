# Machine Learning
# Homework 2
# mapanicci
# krichards

import numpy as np
import matplotlib.pyplot as plt

def loadData(which):
    faces = np.load("age_regression_X{}.npy".format(which))
    faces = faces.reshape(-1, 48, 48)  # Reshape from 576 to 24x24
    labels = np.load("age_regression_y{}.npy".format(which))
    return faces, labels

def makeX(data):
    shape = data.shape
    X = data.reshape(shape[0], shape[1]*shape[2])
    X = np.transpose(X)
    b = np.ones(shape[0])
    X = np.vstack((X, b))


def MSE(y, y_hat):
    e = np.dot(np.transpose(np.subtract(y, y_hat)),np.subtract(y, y_hat))
    return (1/y.shape[0])*e

def solveForW(X, y):
    return np.linalg.solve(np.dot(X, np.transpose(X)), np.dot(X, y))

def makePredictions(X, w):
    return np.dot(np.transpose(X), w)

def testW(trainingFaces, trainingLabels, testingFaces, testingLabels, w, fw):
    X_train = makeX(trainingFaces)
    y_hat_train = makePredictions(X_train, w)
    MSE_train = MSE(trainingLabels, y_hat_train)
    print("Training Error")
    print("MSE:  ", MSE_train)
    print("RMSE: ", np.sqrt(MSE_train))

    X_test = makeX(testingFaces)
    y_hat_test = makePredictions(X_test, w)
    MSE_test = MSE(testingLabels, y_hat_test)
    print("Testing Error")
    print("MSE:  ", MSE_test)
    print("RMSE: ", np.sqrt(MSE_test))
    if fw:
        return findWorst(testingLabels, y_hat_test)

def oneShot(trainingFaces, trainingLabels):
    X = makeX(trainingFaces)
    w = solveForW(X, trainingLabels)
    return w

def getGradient(X, w, y):
    return (1/X.shape[1])*np.dot(X, np.dot(X.T, w) - y)

def getGradient_wReg(X, w, y, a):
    n = X.shape[1]
    gradient = (1/n)*np.dot(X, np.dot(X.T, w) - y)
    reg = (a/n)*w[:-1]
    gradient[:-1] = gradient[:-1] + reg
    return gradient

def gradientDescent(trainingFaces, trainingLabels, wReg):

    # Constants
    learningRate = 0.003
    T = 5000
    std_dev = 0.01
    alpha = 0.1

    # Procedure
    X = makeX(trainingFaces)
    w = std_dev * np.random.randn(X.shape[0])

    # With regularization -- if statement outside of for loops for speed
    if wReg:
        for i in range(T):
            gradient = getGradient_wReg(X, w, trainingLabels, alpha)
            w = w - learningRate*gradient

    # Without regularization
    else:
        for i in range(T):
            gradient = getGradient(X, w, trainingLabels)
            w = w - learningRate*gradient

    return w

def show_w(w):
    newShape = int(np.sqrt(w.shape[0] - 1))
    squareW = np.reshape(w[:-1], (newShape, newShape))

    plt.imshow(squareW)
    plt.show()

def findWorst(y, yhat):
    diff = np.abs(y - yhat)
    top5 = diff.argsort()[::-1][0:5]
    worst = []
    for i in top5:
        worst.append((diff[i], i, y[i], yhat[i]))
    return worst

def displayWorst(worst):
    for img in worst:
        print("\nImage ", img[1])
        print("Predicted Age: ", img[3])
        print("Actual Age:    ", img[2])
        print("Difference: ", img[0])
        plt.imshow(testingFaces[img[1], :, :], cmap='gray')
        plt.show()

if __name__ == "__main__":
    np.random.seed(2)

    trainingFaces, trainingLabels = loadData("tr")
    testingFaces, testingLabels = loadData("te")

    print("\nOne Shot Linear Regression:")
    w1 = oneShot(trainingFaces, trainingLabels)
    testW(trainingFaces, trainingLabels, testingFaces, testingLabels, w1, False)
    show_w(w1)

    print("\nGradient Descent Linear Regression:")
    w2 = gradientDescent(trainingFaces, trainingLabels, False)
    testW(trainingFaces, trainingLabels, testingFaces, testingLabels, w2, False)
    show_w(w2)

    print("\nGradient Descent Linear Regression with Regularization:")
    w3 = gradientDescent(trainingFaces, trainingLabels, True)
    worst = testW(trainingFaces, trainingLabels, testingFaces, testingLabels, w3, True)
    show_w(w3)
    displayWorst(worst)