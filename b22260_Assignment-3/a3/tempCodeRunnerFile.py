def calculate_covmat(X):
    n = X.shape[0]  
    mean = np.mean(X, axis=0)  #mean of each column
    X_sub = X - mean
    C = (X_sub.T @ X_sub)/(n-1)
    return C