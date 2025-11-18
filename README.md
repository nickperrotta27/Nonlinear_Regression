# Nonlinear Regression

## Bias–Variance, Logistic Regression (OvR), and Kernel SVM (OvR)

This repository contains three machine learning experiments implemented from scratch:

- **Bias–Variance analysis** of polynomial regression (1D)
- **Multiclass logistic regression** using One-vs-Rest (OvR) with polynomial features
- **Multiclass RBF-kernel Support Vector Machines** (OvR) using CVXOPT quadratic programming

Each experiment uses a separate dataset, with clean, descriptive filenames (see below).

---

## Part 1 — Polynomial Regression Bias–Variance Analysis

### 📄 Dataset

`poly_bias_variance.pkl`, containing:

```python
{
    "x": np.array([...]),
    "y": np.array([...])
}
```

### 🎯 Goal

Analyze how model complexity (polynomial degree) affects:

- **Bias** (underfitting)
- **Variance** (overfitting)

### 🧠 Method

For degrees 1 through 15:

1. Perform 1000 random 50/50 train–test splits
2. Fit polynomial using `np.polyfit`
3. Track training and test MSE
4. Compute:
   - **Bias** = mean(train error)
   - **Variance** = std(|train error – test error|)

### 📈 Visualization

The notebook produces:

- log(Bias) vs polynomial degree
- log(Variance) vs polynomial degree

This clearly shows U-shaped bias–variance behavior.

---

## Part 2 — Multiclass Logistic Classification (OvR)

### 📄 Dataset

`ovr_classification.pkl`, containing:

```python
{
    "x_train": X_train,   # N × 2
    "y_train": y_train,   # 0,1,2,... classes
    "x_test": X_test,
    "y_test": y_test
}
```

### 🧠 Method

A One-vs-Rest logistic regression classifier is trained:

1. Convert each class `c` into a binary task:
   ```
   y_bin = 1 if y == c else 0
   ```
2. Train using gradient descent on logistic loss
3. Use quadratic polynomial features:
   ```
   [1, x1, x2, x1², x2², x1*x2]
   ```
4. At test time, compute sigmoid score for each class and pick the highest

### 📌 Output

Prints test accuracy of the OvR logistic classifier:

```
Logistic Regression (OvR) Test Accuracy: <value>
```

---

## Part 3 — Multiclass RBF-Kernel SVM (OvR)

### 🧠 Method

A fully custom SVM implementation using:

- RBF kernel
- CVXOPT quadratic programming solver
- One-vs-Rest classification
- Extraction of support vectors and bias term

Prediction using:

$$\text{sign}\left(\sum \alpha_i y_i K(x, x_i) + b\right)$$

### 📌 Steps

For each class `c`:

1. Convert labels to +1 vs -1
2. Solve the dual SVM optimization:
   
   $$\max \sum \alpha_i - \frac{1}{2} \sum\sum \alpha_i \alpha_j y_i y_j K(x_i, x_j)$$
   
   subject to:
   
   $$\sum \alpha_i y_i = 0$$
   $$0 \leq \alpha_i \leq C$$

3. Keep only support vectors

### 📌 Output

Prints:

```
Kernel SVM (OvR) Test Accuracy: <value>
```

---

## Project Structure

```
.
├── poly_bias_variance.pkl        # Part 1 dataset
├── ovr_classification.pkl        # Part 2 + Part 3 dataset
├── bias_variance.ipynb           # Polynomial bias–variance experiment
├── logistic_ovr.ipynb            # Logistic regression (OvR)
├── svm_ovr.ipynb                 # Kernel SVM (OvR)
└── README.md
```

## Dependencies

- `numpy`
- `matplotlib`
- `pickle`
- `cvxopt`

Install locally with:

```bash
pip install numpy matplotlib cvxopt
```

## How to Run

### Google Colab

1. Upload the `.ipynb` files and the dataset files:
   - `poly_bias_variance.pkl`
   - `ovr_classification.pkl`
2. Run all cells

---

## Summary

This repo demonstrates:

- The **bias–variance tradeoff** through polynomial regression
- Building **multiclass logistic regression** from scratch
- Writing your own **multiclass RBF-kernel SVM** using convex optimization
- Understanding **generalization** through repeated random splits
- Comparing **discriminative vs kernel-based classifiers**

These models are implemented **without scikit-learn**, showing all math explicitly.
