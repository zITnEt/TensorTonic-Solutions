import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def grd(w, X, b, y):
    return _sigmoid((X@w+b))-y


def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    # Write code here
    X=np.array(X)
    y=np.array(y)
    w = np.zeros(X.shape[1])
    b=0.0

    for _ in range(steps):
        err = grd(w, X, b, y)
        w -= (X.T@err/len(y))*lr
        b -= lr*err.mean()

    return w, b