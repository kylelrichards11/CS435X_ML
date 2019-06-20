import numpy as np
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import pandas as pd  # only used at very end to show data in pd.DataFrame. Our model is not dependent on pandas


def fPC(y, yhat):
    acc = np.sum((y == yhat)) / y.shape[0]  # compute acc of y vs yhat
    return acc


def measureAccuracyOfPredictors(predictors, X, y):
    pred_bool_sum = np.zeros(X.shape[0])
    for predictor in predictors:  # calculate mean of all yhats
        pred_bool = X[:, predictor[0], predictor[1]] > X[:, predictor[2], predictor[3]]
        pred_bool_sum = pred_bool_sum + pred_bool
    mean_comp = pred_bool_sum / (len(predictors))
    mean_comp = mean_comp > 0.5  # keep as 1 is greater than 0.5, less than or equal to results in no smile
    acc = fPC(y, mean_comp)  # compute acc of y vs yhat (prediction vs labels)
    return acc


def stepwiseRegression(trainingFaces, trainingLabels, testingFaces, testingLabels):
    predictors = []
    max_acc = 0
    for num_feature in range(5):
        predictors.append([])  # add new spot for predictor to predictors arr
        for r1 in range(24):
            for c1 in range(24):
                for r2 in range(24):
                    for c2 in range(24):
                        predictors[num_feature] = [r1, c1, r2, c2]  # fill spot with potential predictor
                        acc = measureAccuracyOfPredictors(predictors, trainingFaces,
                                                          trainingLabels)  # measure accuracy of predictor
                        if acc > max_acc:  # check if new best accuracy
                            predictor_temp = [r1, c1, r2, c2]  # store as temp best predictor
                            max_acc = acc  # use temp best as the new number to beat
        new_predictor = predictor_temp
        predictors[num_feature] = new_predictor  # add best predictor to permanent list of predictors
        test_acc = measureAccuracyOfPredictors(predictors, testingFaces, testingLabels)  # compute testing accuracy
        print("Predictors [[r1,c1,r2,c2]]: " + str(predictors) + "\nTraining Accuracy: " + str(
            max_acc) + "\nTesting Accuracy: " + str(test_acc))  # print predictors, training acc, and testing acc

    return predictors, max_acc, test_acc  # return predictors, training_acc, and testing_acc


def loadData(which):
    faces = np.load("{}ingFaces.npy".format(which))
    faces = faces.reshape(-1, 24, 24)  # Reshape from 576 to 24x24
    labels = np.load("{}ingLabels.npy".format(which))
    return faces, labels

def showFaceImage(predictors):
    # Show an arbitrary test image in grayscale
    im = testingFaces[0, :, :]
    fig, ax = plt.subplots(1)
    ax.imshow(im, cmap='gray')
    color_index = 0
    colors = ['r', 'b', 'g', 'purple', 'orange']
    for p in predictors:
        # Show r1,c1
        rect = patches.Rectangle((p[1] - 0.5, p[0] - 0.5), 1, 1, linewidth=2, edgecolor=colors[color_index],
                                 facecolor='none')
        ax.add_patch(rect)
        # Show r2,c2
        rect = patches.Rectangle((p[3] - 0.5, p[2] - 0.5), 1, 1, linewidth=2, edgecolor=colors[color_index],
                                 facecolor='none')
        ax.add_patch(rect)
        color_index = color_index + 1

    # Display the merged result
    plt.show()

if __name__ == "__main__":

    testingFaces, testingLabels = loadData("test")
    trainingFaces, trainingLabels = loadData("train")

    # Run model for n = 400, 800, 1200, 1600, and 2000 samples
    # produce df with results that was submitted in pdf

    num_samples_arr = []
    training_acc_arr = []
    testing_acc_arr = []
    for n in range(400, 2001, 400):
        print("\n-------------------------------------------------------------------")
        print("TRAINING DATA SIZE", n, "\n")
        predictors, training_acc, testing_acc = stepwiseRegression(trainingFaces[0:n, :, :], trainingLabels[0:n],
                                                                   testingFaces, testingLabels)
        num_samples_arr.append(n)
        training_acc_arr.append(training_acc)
        testing_acc_arr.append(testing_acc)
    results = np.vstack((num_samples_arr, training_acc_arr, testing_acc_arr)).T
    results_df = pd.DataFrame(results, columns=['n', 'trainingAccuracy', 'testingAccuracy'])
    print(results_df)
    show = True
    if show:
        showFaceImage(predictors)