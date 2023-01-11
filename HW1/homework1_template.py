import numpy as np
import matplotlib.pyplot as plt 

def problem_1a (A, B):
    return A + B

def problem_1b (A, B, C):
    return A.dot(B)-C

def problem_1c (A, B, C):
    return A*B + C.T 

def problem_1d (x, y):
    return np.dot(x.T,y)[0][0]

def problem_1e (A, x):
    return np.linalg.solve(A,x)

def problem_1f (A, i):
    return np.sum(A[i].ravel()[::2])

def problem_1g (A, c, d):
    return np.mean(A[np.nonzero((A >= c) & (A <= d))])

def problem_1h (A, k):
    return ...

def problem_1i (x, k, m, s):
    return ...

def problem_1j (A):
    return ...

def problem_1k (x):
    return (x-x.mean())/x.std()

def problem_1l (x, k):
    return np.repeat(x,k,axis=1)

def problem_1m (X, Y):
    return ...

def problem_1n (matrices):
    return ...

def linear_regression (X_tr, y_tr):
    X = X_tr.T 
    y = np.expand_dims(y_tr, axis=1) # transposing y 
    return (np.linalg.inv(X@X.T))@X@y 
def fMSE(X,y,W):
    y = np.expand_dims(y, axis=1)
    return np.sum((X@W-y)**2)/(2*X.shape[0])  #np.sum((X@W- y)**2)/(2*X.shape[0])

def add_bias(X):
    return np.hstack((X,np.ones((X.shape[0],1))))

def plot(y,y_hat,label="Training"):
    plt.xlabel("Index")
    plt.ylabel("Age")
    plt.title(label)
    plt.plot(y)
    plt.plot(y_hat)
    plt.legend(['Original', 'Predicted'], loc='upper left')
    plt.show()

def train_age_regressor ():
    # Load data
    X_tr = np.reshape(np.load("age_regression_Xtr.npy"), (-1, 48*48))
    ytr = np.load("age_regression_ytr.npy")
    X_te = np.reshape(np.load("age_regression_Xte.npy"), (-1, 48*48))
    yte = np.load("age_regression_yte.npy")
    # Adding bias term for all the train and test data sets 
    X_tr = add_bias(X_tr)
    X_te = add_bias(X_te) 
    W = linear_regression(X_tr, ytr)
    ytr_hat = X_tr@W
    yte_hat = X_te@W 
    plot(ytr,ytr_hat,label = "Training")
    plot(yte,yte_hat,label = "Testing")

    # Report fMSE cost on the training and testing data (separately)
    # ...
    train_err = fMSE(X_tr,ytr,W)
    test_err   = fMSE(X_te,yte,W)
    print("Training Error", train_err)
    print("Test Error" , test_err)

if __name__ == "__main__":
    train_age_regressor()