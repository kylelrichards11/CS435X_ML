# Machine Learning Homework 5
# mapanicci
# krichards

import numpy as np
import matplotlib.pyplot as plt

########################################################################################################################

def loadData(which):
    digits = np.load("small_mnist_{}_images.npy".format(which))
    digits = digits.reshape(-1, 28, 28)
    labels = np.load("small_mnist_{}_labels.npy".format(which))
    return digits, labels

def makeX(data):
    shape = data.shape
    X = data.reshape(shape[0], shape[1]*shape[2])
    return np.transpose(X)

def PCA(X):
    x_bar = np.mean(X, axis=1)
    X_tilde = (X.T - x_bar.T).T
    X_sqr = np.dot(X_tilde, X_tilde.T)
    e_vals, e_vecs = np.linalg.eig(X_sqr)
    pc1 = e_vecs[:, 0]
    pc2 = e_vecs[:, 1]
    p1 = X_tilde.T.dot(pc1)
    p2 = X_tilde.T.dot(pc2)
    return p1, p2

########################################################################################################################
# Main

if __name__ == "__main__":
    testingDigits, testingLabels = loadData("test")
    X = makeX(testingDigits)
    p1, p2 = PCA(X)

    # Plot with labels
    colors = testingLabels.argmax(axis=1)
    plt.scatter(p1, p2, s=3, c=colors, cmap='jet')
    plt.show()
