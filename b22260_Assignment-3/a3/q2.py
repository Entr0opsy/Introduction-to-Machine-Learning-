import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from scipy.stats import multivariate_normal

def mean(arr):
    sum = 0
    for i in arr:
        sum += i
    return sum/len(arr)

def calculate_covmat(X):
    n = X.shape[0]  
    mean = np.mean(X, axis=0)  #mean of each column
    X_sub = X - mean
    C = (X_sub.T @ X_sub)/(n-1)
    return C

def likelihood(X, X_mean, C):
    mvn = multivariate_normal(mean = X_mean, cov = C)
    p = mvn.pdf(X)
    return p

def get_parameters(train, y):
    mean_vectors = np.zeros((3,4)) 
    covm = []
    prior = [] 

    for i in range(0, 3):  
        for j in range(0, 4):   
            values = train[train['Species'] == y[i]][attributes[j]].values   
            mean_vectors[i][j] = mean(values) 

    for i in range(0,3):
        data = train[train['Species'] == y[i]].to_numpy()[:, 1:-1]
        C = calculate_covmat(data)
        covm.append(C)
        prior.append(len(data)/len(train))

    return mean_vectors, covm, prior

def bayes_multivar(train_df, test_df): 
    mean_vectors, covm, prior = get_parameters(train_df, y)

    predicted = []
    test_matrix = test_df.to_numpy()[:,1:-1]

    for sample in test_matrix:
        posterior = []
        posterior.append(prior[0]*(likelihood(sample, mean_vectors[0], covm[0])))
        posterior.append(prior[1]*(likelihood(sample, mean_vectors[1], covm[1])))
        posterior.append(prior[2]*(likelihood(sample, mean_vectors[2], covm[2])))
        index = np.argmax(posterior)
        predicted.append(y[index])

    return predicted

def conf_matrix(test_class, predicted):
    cm = metrics.confusion_matrix(test_class, predicted)
    cm_display = metrics.ConfusionMatrixDisplay(cm, display_labels = y)
    cm_display.plot()
    plt.title("Confusion Matrix")
    return cm

def accuracy(conf_mat):
    n = len(conf_mat)
    correct = 0
    total = 0
    for i in range(0, n):
        for j in range(0, n):
            if(i == j):
                correct += conf_mat[i][j]
            total += conf_mat[i][j]

    return (correct/total)*100

train_df = pd.read_csv('iris_train.csv')
test_df = pd.read_csv('iris_test.csv')

attributes = train_df.columns[1:-1] 
y_test = test_df['Species'].tolist()
y_train = train_df['Species'].tolist()
y = train_df['Species'].unique() 

y_pred = bayes_multivar(train_df, test_df)
cm = conf_matrix(y_test, y_pred)
plt.show()

print("Confusion Matrix:\n",cm)
print("\nAccuracy =", accuracy(cm), "%")