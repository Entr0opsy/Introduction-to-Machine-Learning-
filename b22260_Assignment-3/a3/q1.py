import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn import metrics
import assignment2

test_df = pd.read_csv("iris_test.csv")
train_df = pd.read_csv("iris_train.csv")

y_test = test_df["Species"].tolist() 
y_train = train_df["Species"].tolist()

#print(y_test)
X_test = test_df.to_numpy()[:, 1:-1]
X_train = train_df.to_numpy()[:, 1:-1]

reduced_X_test = assignment2.pca(X_test.T, 1)
reduced_X_train = assignment2.pca(X_train.T, 1)
print(reduced_X_train)

def mean(arr):
    sum = 0
    for i in arr:
        sum += i
    return sum/len(arr)

def std(numbers):
    avg = mean(numbers)
    variance = sum([(x-avg)**2 for x in numbers]) / float(len(numbers)-1)
    return math.sqrt(variance)

def calc(x, mean, var, prior_prob):
    val = -1*((x-mean)**2)/(2*var)
    den = 1/(2*math.pi*var)**0.5
    return prior_prob*den*(math.exp(val))

train_df = pd.DataFrame(reduced_X_train, columns=['data'])
train_df['Species'] = y_train
test_df =pd.DataFrame(reduced_X_test, columns=['data'])
test_df['Species'] = y_test

y = train_df['Species'].unique()

def get_parameters(train_df, y):
    means = []
    variances = []
    prior = []

    for i in range(0, 3):
        list = train_df[train_df['Species'] == y[i]]['data'].values
        means.append(sum(list)/len(list))
        variances.append(sum((x - means[i])**2 for x in list) / len(list))
        prior.append(len(list)/120)
    
    return means, variances, prior
    
def bayes_univar(train_df, reduced_X_test):
    means, variances, prior = get_parameters(train_df, y)
    predicted = []
    for sample in reduced_X_test:
        posterior = []
        posterior.append(calc(sample, means[0], variances[0], prior[0]))
        posterior.append(calc(sample, means[1], variances[1], prior[1]))
        posterior.append(calc(sample, means[2], variances[2], prior[2]))
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

y_pred = bayes_univar(train_df, reduced_X_test)
cm = conf_matrix(y_test, y_pred)
plt.show()

print("Confusion Matrix:\n", cm)
print("\nAccuracy =", accuracy(cm), "%")