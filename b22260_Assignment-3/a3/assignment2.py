import numpy as np

def subtractMeans(data):
    mean = np.mean(data)
    output = [i-mean for i in data]
    return output

def pca(X, n): # (the rows of X are the attributes)
    X_sub = []
    for attribute in X:
        attribute = subtractMeans(attribute)
        X_sub.append(attribute) 

    X = np.array(X)
    X = X.T 

    X_sub = np.array(X_sub)  # X_sub is the mean subtracted data of X
    X_sub = np.transpose(X_sub) 

    C = np.matmul(X_sub.T, X_sub) 

    eigenvalues, eigenvectors = np.linalg.eig(C)

    # sorting the eigenvalues and corresponding eigenvectors in descending order
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]

    Q = eigenvectors[:, :n].T  
    X_red = np.matmul(X_sub, Q.T) 
    return X_red
