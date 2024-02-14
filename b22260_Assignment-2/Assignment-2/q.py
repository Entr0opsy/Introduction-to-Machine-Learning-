import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
import statistics as stat
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


df = pd.read_csv('./Iris.csv')
matrix = (df.iloc[:, :4].values)
y = (df.iloc[:, 4].values)
indep = ["SepalLengthCm"  ,"SepalWidthCm"  ,"PetalLengthCm","PetalWidthCm"]


def is_replace(x, q1, q3, iqr):
    if (q1 - 1.5 * iqr > x or q3 + 1.5 * iqr < x):
        return True
    return False

def iqr(l):
    l.sort()
    return(math.fabs(l[int(len(l)/4)] - l[int((3*len(l))/4)]))
def q1(l):
    l.sort()
    return(l[int(len(l)/4)])
def q3(l):
    l.sort()
    return(l[int((3*len(l))/4)])

for col in indep:
    q1_val = q1(df[col].tolist())
    q3_val = q3(df[col].tolist())
    iqr_val = iqr(df[col].tolist())
    o = 0
    for i in range(len(df[col])):
        if is_replace(df.at[i,col], q1_val, q3_val, iqr_val):
            o+=1
            df.at[i, col] = np.median(df[col].tolist())

l = []
for i in indep:
    l.append(np.mean(df[i]))
l2 = []
for i in matrix:
    l2.append(i - np.array(l))
l2 = np.array(l2)


matrix = l2
c =  np.matmul(matrix.T,matrix)

eigenvalues, eigenvectors = np.linalg.eig(c)

# eigen_dict = {eigenvalues[i]: eigenvectors[:, i] for i in range(len(eigenvalues))}
# sorted_eigen_dict = dict(sorted(eigen_dict.items()))
# print(c)

eigenvectors = pd.DataFrame(eigenvectors)
#print(eigenvectors)
first_key_value = eigenvectors.iloc[:, 0].values
second_key_value = eigenvectors.iloc[:, 1].values

print(first_key_value,second_key_value)
qmatrix = np.row_stack(((first_key_value), (second_key_value)))
xcap = np.matmul(matrix,qmatrix.T)
dxcap = pd.DataFrame(xcap)

plt.scatter(dxcap.iloc[:, 0].values,dxcap.iloc[:, 1].values)
# Plot eigendirection 1
plt.quiver(0, 0, qmatrix[0][0], qmatrix[0][1], angles='xy', scale_units='xy', scale=1, color='red', label='Eigendirection 1')

# Plot eigendirection 2
plt.quiver(0, 0, qmatrix[1][0], qmatrix[1][1], angles='xy', scale_units='xy', scale=1, color='blue', label='Eigendirection 2')


plt.show()

reconstructeddata = np.matmul(dxcap,qmatrix)    
print("Reconstructed data is")
print(reconstructeddata)



squared_errors = (reconstructeddata - matrix)**2

# Calculate the mean squared error (MSE)
mse = np.mean(squared_errors)

# Calculate the root mean square error (RMSE)
rmse = mse**0.5

print("Root Mean Square Error (RMSE):", rmse)




#Question2 begins here


X_train, X_test, y_train, y_test = train_test_split(xcap, y, random_state=104, test_size=0.20, shuffle=True)

print(y_test)
k = 5

def mostfreq(d,k):
    l = np.array(d[0:k])[:,1]
    return stat.mode(l)


yt = []
for xitest in X_test: #changing X_test to xcap
    
    ind2 = 0
    d = []
    for xjtrain in X_train:
        d.append([np.linalg.norm(xitest-xjtrain),y_train[ind2]])
        ind2+=1        
    d.sort(key = lambda d:d[0])
   # yt = mostfreq(sorted_dict,k)
    yt.append(mostfreq(d,k))
    
cm = confusion_matrix(y_test, np.array(yt))
print(pd.DataFrame(yt,y_test))
# Create a figure and axis for the plot
disp = ConfusionMatrixDisplay(confusion_matrix=cm)

# Plot the confusion matrix
disp.plot(cmap='viridis')

# Add a title (optional)
plt.title("Confusion Matrix")

# Show the plot
plt.show()