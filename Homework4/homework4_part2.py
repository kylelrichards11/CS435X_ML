# Machine Learning HW4 Part 2
# mapanicci and krichards

## RESULTS
# Linear AUC:	    0.8586373868481646
# Polynomial AUC:   0.8417210302948643

import sklearn.svm as svm
import sklearn.metrics
import numpy as np
import pandas
from sklearn.model_selection import train_test_split

# Load data
print("\nLoading Data")
d = pandas.read_csv('train.csv')
y = np.array(d.target)  # Labels
X = np.array(d.iloc[:,2:])  # Features

# Split into train/test folds
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, shuffle=True)

# Linear SVM
print("\nRunning Linear SVM")
lin_clf = svm.LinearSVC(dual=False)
lin_clf.fit(X_train, y_train)

# Bagging
y_train = y_train.reshape(y_train.shape[0],1)
full_arr = np.hstack((X_train,y_train))
splits = 20
full_arr_split = np.split(full_arr,splits)

# Non-linear SVM (polynomial kernel)
print("\nRunning Polynomial SVM")
y_hat_poly = np.zeros((y_train.shape[0],1))
j = 0
for i in full_arr_split:
  j = j + 1
  X_train = i[:,:-1]
  y_train = i[:,-1]
  clf = svm.SVC(kernel = 'poly', gamma='auto')
  clf.fit(X_train, y_train)
  y_hat_poly_it = clf.decision_function(X_test).reshape(y_hat_poly.shape[0],1)
  print("Poly iteration ", j)
  y_hat_poly = y_hat_poly + y_hat_poly_it


# Apply the SVMs to the test set
y_hat_lin = lin_clf.decision_function(X_test) # Linear kernel

# Compute AUC
auc_lin = sklearn.metrics.roc_auc_score(y_test, y_hat_lin)
auc_poly = sklearn.metrics.roc_auc_score(y_test, y_hat_poly/splits)

print("\nLinear AUC:\t\t", auc_lin)
print("Polynomial AUC:\t", auc_poly)