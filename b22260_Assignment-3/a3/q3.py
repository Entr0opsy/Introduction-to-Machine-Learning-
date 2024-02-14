import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn import metrics
from scipy.stats import multivariate_normal
from assignment2 import pca
from q1 import bayes_univar, conf_matrix, accuracy
from q2 import bayes_multivar

# 1-dimensional data (reduced)

train_df = pd.read_csv('iris_train.csv')
test_df = pd.read_csv('iris_test.csv')

y_test = test_df["Species"].tolist() 
y_train = train_df["Species"].tolist()

X_test = test_df.to_numpy()[:, 1:-1]
X_train = train_df.to_numpy()[:, 1:-1]

reduced_X_test = pca(X_test.T, 1)
reduced_X_train = pca(X_train.T, 1)

train_df = pd.DataFrame(reduced_X_train, columns=['data'])
train_df['Species'] = y_train
test_df = pd.DataFrame(reduced_X_test, columns=['data'])
test_df['Species'] = y_test

pred1 = bayes_univar(train_df, reduced_X_test)


# 4-dimensional data (original)

train_df = pd.read_csv('iris_train.csv')
test_df = pd.read_csv('iris_test.csv')

pred2 = bayes_multivar(train_df, test_df)

cm1 = conf_matrix(y_test, pred1)
cm2 = conf_matrix(y_test, pred2)

acc1 = accuracy(cm1)
acc2 = accuracy(cm2)

print("Difference in accuracy =", acc2-acc1)