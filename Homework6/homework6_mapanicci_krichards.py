import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import scipy.optimize
import pandas as pd
from sklearn.decomposition import PCA

pd.set_option('display.max_rows', 31900)
pd.set_option('display.max_columns', 30000)
pd.set_option('display.width', 1000)

NUM_INPUT = 784  # Number of input neurons
NUM_HIDDEN = 40  # Number of hidden neurons
NUM_OUTPUT = 10  # Number of output neurons
NUM_CHECK = 5  # Number of examples on which to check the gradient
LEARNING_RATE = 0.1
EPOCHS = 15
BATCH_SIZE = 2500
TEST_HYPER = True
USE_BEST_HYPER = True

# Given a vector w containing all the weights and biased vectors, extract
# and return the individual weights and biases W1, b1, W2, b2.
# This is useful for performing a gradient check with check_grad.
def unpack(w):
    a = NUM_HIDDEN*NUM_INPUT
    b = a + NUM_HIDDEN
    c = b + NUM_HIDDEN*NUM_OUTPUT
    d = c + NUM_OUTPUT
    splitW = np.split(w, [a, b, c, d])[0:4]
    W1 = splitW[0].reshape(NUM_INPUT, NUM_HIDDEN)
    b1 = splitW[1].T
    W2 = splitW[2].reshape(NUM_HIDDEN, NUM_OUTPUT)
    b2 = splitW[3].T
    return W1, b1, W2, b2


# Given individual weights and biases W1, b1, W2, b2, concatenate them and
# return a vector w containing all of them.
# This is useful for performing a gradient check with check_grad.
def pack(x1, x2, x3, x4):
    return np.concatenate((x1.flatten(), x2.flatten(), x3.flatten(), x4.flatten()))


# Load the images and labels from a specified dataset (train or test).
def loadData(which):
    images = np.load("mnist_{}_images.npy".format(which))
    labels = np.load("mnist_{}_labels.npy".format(which))
    return images, labels


def plotSGDPath(trainX, trainY, w):
    trainX = trainX[0:2500, :]
    trainY = trainY[0:2500, :]

    global LEARNING_RATE
    LEARNING_RATE = 0.25
    EPOCHS = 100

    CEs = []
    Ws = []
    for epoch in range(EPOCHS):
        yhat = forwardProp(trainX, w)
        w = backProp(trainX, trainY, w, yhat)
        ce, pc = getErrors(trainX, trainY, w, yhat)
        CEs.append(ce)
        Ws.append(w.tolist())

    Ws = np.array(Ws)

    pca = PCA(n_components=2, svd_solver="full")
    pca_w = pca.fit_transform(Ws)

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("CE")
    ax.scatter(pca_w[:, 0], pca_w[:, 1], np.array(CEs), color='r')

    # Compute the CE loss on a grid of points (corresonding to different w).
    axis1 = np.arange(-3, 5, 0.1)
    axis2 = np.arange(-3, 5, 0.1)

    Xaxis, Yaxis = np.meshgrid(axis1, axis2)
    Zaxis = np.zeros((len(axis1), len(axis2)))

    for i in range(len(axis1)):
        for j in range(len(axis2)):
            w_inv = pca.inverse_transform(np.array([Xaxis[i, j], Yaxis[i, j]]))
            Zaxis[i, j] = fCE(trainX, trainY, w_inv)
    ax.plot_surface(Xaxis, Yaxis, Zaxis, alpha=0.6)  # Keep alpha < 1 so we can see the scatter plot too.

    plt.show()


def percentCorrect(y, yhat):
    return sum(np.argmax(yhat, axis=1) == np.argmax(y, axis=1)) / yhat.shape[0]


# Given training images X, associated labels Y, and a vector of combined weights
# and bias terms w, compute and return the cross-entropy (CE) loss. You might
# want to extend this function to return multiple arguments (in which case you
# will also need to modify slightly the gradient check code below).
def fCE(X, y, w):
    yhat = forwardProp(X, w)
    return -np.sum(y * np.log(yhat)) / y.shape[0]


# Given training images X, associated labels Y, and a vector of combined weights
# and bias terms w, compute and return the gradient of fCE. You might
# want to extend this function to return multiple arguments (in which case you
# will also need to modify slightly the gradient check code below).
def gradCE(X, y, w, yhat=None):
    W1, b1, W2, b2 = unpack(w)
    if yhat is None:
        yhat = forwardProp(X, w)
    z1 = np.dot(X, W1) + b1  # Switched X and W1
    g = (np.dot(W2, (yhat - y).T) * relu_der(z1).T).T  # Switched W2 and yhat-y
    h1 = relu(z1)

    W1_grad = np.dot(X.T, g) / X.shape[0]  # Switched X and g, made
    b1_grad = np.mean(g, axis=0)
    W2_grad = np.dot(h1.T, (yhat - y)) / X.shape[0]  # Switched h1.T and yhat-y
    b2_grad = np.mean((yhat - y), axis=0)

    grad = pack(W1_grad, b1_grad, W2_grad, b2_grad)

    return grad


# Given training and testing datasets and an initial set of weights/biases b,
# train the NN. Then return the sequence of w's obtained during SGD.
def train(trainX, trainY, w, plot=False, p=True, latex=False):
    if plot:
        plotVals = []

    if p:
        print()

    for epoch in range(EPOCHS):
        for i in range(0, trainX.shape[0], BATCH_SIZE):
            trainX_mb = trainX[i:i + BATCH_SIZE, :]
            trainY_mb = trainY[i:i + BATCH_SIZE, :]
            yhat = forwardProp(trainX_mb, w)
            w = backProp(trainX_mb, trainY_mb, w, yhat)
        yhat = forwardProp(trainX, w)
        ce, pc = getErrors(trainX, trainY, w, yhat)
        if plot:
            plotVals.append([epoch, ce, pc])
        if p:
            print("Epoch:", epoch, " CE:", round(ce, 4), " PC:", round(pc, 4))

    if plot:
        descent = pd.DataFrame(np.array(plotVals), columns=["Epoch", "Cross Entropy", "Percent Correct"])
        if latex:
            saveLatexTable(descent)
        plt.xlabel("Epochs")
        plt.ylabel("Cross Entropy")
        plt.plot(descent["Epoch"], descent["Cross Entropy"])
        plt.show()
    return w


def getErrors(X, y, w, yhat):
    ce = round(fCE(X, y, w), 4)
    pc = percentCorrect(y, yhat)
    return ce, pc


def relu(a):
    b = np.copy(a)
    b[b < 0] = 0
    return b


def relu_der(a):
    b = np.copy(a)
    b[b >= 0] = 1
    b[b < 0] = 0
    return b


def softmax(yhat):
    yhat = np.exp(yhat)
    yhatSum = np.sum(yhat, axis=1).reshape(-1, 1)
    return yhat / yhatSum


def forwardProp(X, w):
    W1, b1, W2, b2 = unpack(w)
    z1 = np.dot(X, W1) + b1  # switched X and W1
    h1 = relu(z1)
    z2 = np.dot(h1, W2) + b2  # switched h1 and W2
    return softmax(z2)


def backProp(X, y, w, yhat):
    g = gradCE(X, y, w, yhat=yhat)
    return changeW(g, w)


def changeW(grad, w):
    W1, b1, W2, b2 = unpack(w)
    g_w1, g_b1, g_w2, g_b2 = unpack(grad)
    W1 = W1 - LEARNING_RATE * g_w1
    b1 = b1 - LEARNING_RATE * g_b1
    W2 = W2 - LEARNING_RATE * g_w2
    b2 = b2 - LEARNING_RATE * g_b2
    return pack(W1, b1, W2, b2)


def shuffleData(X, y):
    X_y = np.hstack((X, y))
    np.random.shuffle(X_y)
    return X_y[:, :-10], X_y[:, -10:]


def test(testX, testY, w, p=True):
    yhat = forwardProp(testX, w)
    ce, pc = getErrors(testX, testY, w, yhat)
    if p:
        print("TESTING ERRORS:  CE:", round(ce, 4), " PC:", round(pc, 4))
    return ce, pc

def saveLatexTable(dataframe):
    fileName = "tables.txt"
    with open(fileName, "a") as file:
        file.write(dataframe.to_latex())


def findBestHyperparameters(trainX, trainY, valX, valY, latex=False):
    print("FINDING BEST HYPERPARAMETERS")
    global NUM_HIDDEN, LEARNING_RATE, BATCH_SIZE
    hiddenNeurons = [40, 50]
    learningRates = [0.1, 0.25, 0.5]
    batchSizes = [50, 100]
    out = []
    out_pd = None
    for h in hiddenNeurons:
        for l in learningRates:
            for b in batchSizes:
                NUM_HIDDEN = h
                LEARNING_RATE = l
                BATCH_SIZE = b
                ce, pc, _ = runNetwork(trainX, trainY, valX, valY, p=True)
                out.append([h, l, b, ce, pc])
                out_pd = pd.DataFrame(np.array(out),
                                   columns=["Hidden Neurons", "Learning Rate", "Batch Size", "Cross Entropy",
                                            "Percent Correct"])
                print()
                print(out_pd)

    best = out_pd.sort_values(by="Cross Entropy").iloc[0, :].values[0:3]

    if latex:
        saveLatexTable(out_pd.sort_values(by="Cross Entropy"))

    print("\nBest Hyperparameters:")
    print("Hidden Neurons:", best[0].astype(int), "\nLearning Rate:", best[1], "\nBatch Size:", best[2].astype(int))
    NUM_HIDDEN, LEARNING_RATE, BATCH_SIZE = best[0].astype(int), best[1], best[2].astype(int)


def runNetwork(trainX, trainY, newX, newY, p=True, plot=False, latex=False):
    w = randomWeights()
    ws = train(trainX, trainY, w, p=p, plot=plot, latex=latex)
    ce, pc = test(newX, newY, ws, p=p)
    return ce, pc, ws

def randomWeights():
    # Initialize weights randomly
    W1 = 2 * (np.random.random(size=(NUM_INPUT, NUM_HIDDEN)) / NUM_INPUT ** 0.5) - 1. / NUM_INPUT ** 0.5
    b1 = 0.01 * np.ones(NUM_HIDDEN)
    W2 = 2 * (np.random.random(size=(NUM_HIDDEN, NUM_OUTPUT)) / NUM_HIDDEN ** 0.5) - 1. / NUM_HIDDEN ** 0.5
    b2 = 0.01 * np.ones(NUM_OUTPUT)

    return pack(W1, b1, W2, b2)


if __name__ == "__main__":
    np.random.seed(2)

    # Load data
    trainX, trainY = loadData("train")
    valX, valY = loadData("validation")
    testX, testY = loadData("test")

    trainX, trainY = shuffleData(trainX, trainY)

    w = randomWeights()

    # Check that the gradient is correct on just a few examples (randomly drawn).
    idxs = np.random.permutation(trainX.shape[0])[0:NUM_CHECK]
    print("CHECK GRAD")
    print(scipy.optimize.check_grad(lambda w_: fCE(np.atleast_2d(trainX[idxs, :]), np.atleast_2d(trainY[idxs, :]), w_),
                                    lambda w_: gradCE(np.atleast_2d(trainX[idxs, :]), np.atleast_2d(trainY[idxs, :]),
                                                      w_), w))
    print()


    if TEST_HYPER:
        findBestHyperparameters(trainX, trainY, valX, valY, latex=True)

    if True:
        print("\nRunning with best hyperparameters:")
        EPOCHS = 40
        ce, pc, w = runNetwork(trainX, trainY, testX, testY, plot=True, latex=True)
        print("\nFINAL RESULTS")
        print("CE:", ce, "PC:", pc)


    # Plot the SGD trajectory
    # plotSGDPath(trainX, trainY, ws)
    w = randomWeights()
    print("\n Visualizing SGD")
    plotSGDPath(trainX, trainY, w)
