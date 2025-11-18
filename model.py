from google.colab import files

uploaded = files.upload()
print(type(data))
print(data.keys() if hasattr(data, "keys") else None)
print(data[:5] if isinstance(data, (list, tuple)) else None)

import pickle
import numpy as np
import matplotlib.pyplot as plt

# Load data
with open("poly_bias_variance.pkl", "rb") as f:
    data = pickle.load(f)

x = data["x"]
y = data["y"]

def fit_polynomial(x_train, y_train, degree):
    # Fit polynomial coefficients using least squares
    coeffs = np.polyfit(x_train, y_train, degree)
    return coeffs

def predict_polynomial(coeffs, x):
    # Evaluate polynomial with given coefficients
    return np.polyval(coeffs, x)

def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

degrees = range(1, 16)
n_runs = 1000

train_errors = {d: [] for d in degrees}
test_errors = {d: [] for d in degrees}

for _ in range(n_runs):
    # random 50/50 split
    idx = np.arange(len(x))
    np.random.shuffle(idx)
    split = len(x) // 2
    train_idx, test_idx = idx[:split], idx[split:]

    x_train, y_train = x[train_idx], y[train_idx]
    x_test, y_test = x[test_idx], y[test_idx]

    for d in degrees:
        coeffs = fit_polynomial(x_train, y_train, d)

        y_train_pred = predict_polynomial(coeffs, x_train)
        y_test_pred = predict_polynomial(coeffs, x_test)

        train_errors[d].append(mse(y_train, y_train_pred))
        test_errors[d].append(mse(y_test, y_test_pred))

# Bias: mean training error
bias = [np.mean(train_errors[d]) for d in degrees]

# Variance: std of |train error - test error|
variance = [np.std(np.abs(np.array(train_errors[d]) - np.array(test_errors[d]))) for d in degrees]
plt.plot(degrees, np.log(bias), label="log(Bias)")
plt.plot(degrees, np.log(variance), label="log(Variance)")
plt.xlabel("Polynomial Degree")
plt.ylabel("Log Error")
plt.legend()
plt.show()

import numpy as np
import pickle

# Load dataset
with open("ovr_classification.pkl", "rb") as f:
    data = pickle.load(f)

X_train, y_train = data["x_train"], data["y_train"]
X_test, y_test = data["x_test"], data["y_test"]

if X_train.ndim == 1:
    X_train = X_train.reshape(-1, 1)
    X_test = X_test.reshape(-1, 1)

# Polynomial basis (degree=2)
def poly_features(X):
    x1, x2 = X[:, 0], X[:, 1]
    return np.vstack([np.ones(len(X)), x1, x2, x1**2, x2**2, x1*x2]).T

# Sigmoid
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Binary logistic regression training
def train_logistic_binary(X, y, lr=0.1, epochs=2000):
    X_poly = poly_features(X)
    w = np.zeros(X_poly.shape[1])
    for _ in range(epochs):
        preds = sigmoid(X_poly @ w)
        grad = X_poly.T @ (preds - y) / len(y)
        w -= lr * grad
    return w

# Train OvR logistic regression
def train_ovr(X, y):
    classes = np.unique(y)
    models = {}
    for c in classes:
        y_bin = (y == c).astype(int)  # 1 for class c, 0 otherwise
        w = train_logistic_binary(X, y_bin)
        models[c] = w
    return models

# Predict with OvR
def predict_ovr(X, models):
    X_poly = poly_features(X)
    scores = []
    for c, w in models.items():
        scores.append(sigmoid(X_poly @ w))
    scores = np.vstack(scores).T
    return np.argmax(scores, axis=1)

# Train OvR models
models = train_ovr(X_train, y_train)

# Predict on test set
y_pred = predict_ovr(X_test, models)

# Accuracy
acc_log = np.mean(y_pred == y_test)
print("Logistic Regression (OvR) Test Accuracy:", acc_log)

from cvxopt import matrix, solvers
import numpy as np

# RBF Kernel
def rbf_kernel(X1, X2, sigma=0.5):
    sq_dists = np.sum(X1**2, axis=1).reshape(-1, 1) + np.sum(X2**2, axis=1) - 2 * X1 @ X2.T
    return np.exp(-sq_dists / (2 * sigma**2))

# Train binary SVM
def train_svm(X, y, C=1.0, sigma=0.5):
    n = X.shape[0]
    K = rbf_kernel(X, X, sigma)

    P = matrix(np.outer(y, y) * K)
    q = matrix(-np.ones(n))
    G = matrix(np.vstack([-np.eye(n), np.eye(n)]))
    h = matrix(np.hstack([np.zeros(n), np.ones(n) * C]))
    A = matrix(y.astype(float), (1, n))
    b = matrix(0.0)

    solvers.options['show_progress'] = False
    sol = solvers.qp(P, q, G, h, A, b)
    alphas = np.ravel(sol['x'])

    sv = alphas > 1e-5
    alpha_sv, X_sv, y_sv = alphas[sv], X[sv], y[sv]

    # Fix: use only the support vector kernel block
    K_sv = K[np.ix_(sv, sv)]
    b_val = np.mean([y_sv[i] - np.sum(alpha_sv * y_sv * K_sv[i]) for i in range(len(alpha_sv))])

    return alpha_sv, X_sv, y_sv, b_val

# Predict binary SVM
def predict_svm(X_sv, y_sv, alpha_sv, b, X_test, sigma=0.5):
    K = rbf_kernel(X_test, X_sv, sigma)
    return np.sign(K @ (alpha_sv * y_sv) + b)
# Train OvR multi-class SVM
def train_multiclass_svm(X, y, C=1.0, sigma=0.5):
    classes = np.unique(y)
    models = {}
    for c in classes:
        y_bin = np.where(y == c, 1, -1)
        alpha_sv, X_sv, y_sv, b = train_svm(X, y_bin, C, sigma)
        models[c] = (alpha_sv, X_sv, y_sv, b)
    return models

# Predict OvR multi-class SVM
def predict_multiclass_svm(models, X_test, sigma=0.5):
    scores = []
    for c, (alpha_sv, X_sv, y_sv, b) in models.items():
        K = rbf_kernel(X_test, X_sv, sigma)
        score = K @ (alpha_sv * y_sv) + b
        scores.append(score)
    scores = np.vstack(scores).T
    return np.argmax(scores, axis=1)

# Train OvR SVM
models = train_multiclass_svm(X_train, y_train, C=1.0, sigma=0.5)

# Predict
y_pred_svm = predict_multiclass_svm(models, X_test, sigma=0.5)

# Accuracy
acc_svm = np.mean(y_pred_svm == y_test)
print("Kernel SVM (OvR) Test Accuracy:", acc_svm)
