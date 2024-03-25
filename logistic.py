import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression


class CustomLogisticRegression:

    def __init__(self, fit_intercept: bool = True, l_rate=0.01, n_epoch=100):
        self.fit_intercept = fit_intercept
        self.l_rate = l_rate
        self.n_epoch = n_epoch
        self.coef_ = None

    @staticmethod
    def sigmoid(t):
        return 1 / (1 + np.exp(-t))

    def predict_proba(self, row: np.array(float), coef_: np.array(float)):
        if self.fit_intercept:
            t = coef_[0] + row.dot(coef_[1:])
        else:
            t = row.dot(coef_)
        return self.sigmoid(t)

    def fit(self, X: pd.DataFrame, y: pd.Series, cost: str = 'mse'):
        X, y = X.to_numpy(), y.to_numpy()
        # cost function and its gradient
        if cost == 'log_loss':
            def error():
                y1 = self.predict_proba(X, self.coef_)
                return -1 * (y * np.log(y1) + (1 - y) * np.log(1 - y1)).mean()

            def grad(y0, y1):
                return (y1 - y0) / X.shape[0]
        else:  # mse
            def error():
                return np.square(self.predict_proba(X, self.coef_) - y).mean()

            def grad(y0, y1):
                return (y1 - y0) * y1 * (1 - y1)
        # initial weights
        if self.fit_intercept:
            self.coef_ = np.zeros(X.shape[1] + 1)
        else:
            self.coef_ = np.zeros(X.shape[1])
        error_first = []
        error_last = []

        for epoch in range(self.n_epoch):
            for i, row in enumerate(X):
                y_hat = self.predict_proba(row, self.coef_)
                k = self.l_rate * grad(y[i], y_hat)
                if self.fit_intercept:
                    self.coef_[0] -= k
                    self.coef_[1:] -= k * row
                else:
                    self.coef_ -= k * row
                if epoch == 0:
                    error_first.append(error())
                if epoch == self.n_epoch - 1:
                    error_last.append(error())
        return error_first, error_last

    def fit_mse(self, X: pd.DataFrame, y: pd.Series):
        return self.fit(X, y, cost='mse')

    def fit_log_loss(self, X: pd.DataFrame, y: pd.Series):
        return self.fit(X, y, cost='log_loss')

    def predict(self, data: pd.Series, cut_off: float = 0.5):
        data = data.to_numpy()
        if self.coef_ is None:
            raise AttributeError('Model is not fit')
        y_hat = self.predict_proba(data, self.coef_)
        predictions = [0 if p < cut_off else 1 for p in y_hat]
        return np.array(predictions)


features = ['worst concave points', 'worst perimeter', 'worst radius']
X, y = load_breast_cancer(return_X_y=True, as_frame=True)
X = X[features]
X = (X - X.mean()) / X.std()  # standardize
lr = CustomLogisticRegression(fit_intercept=True, l_rate=0.01, n_epoch=1000)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=43)

out = dict()

mse_err = lr.fit_mse(X_train, y_train)
out['mse_accuracy'] = accuracy_score(y_test, lr.predict(X_test))

logloss_err = lr.fit_log_loss(X_train, y_train)
out['logloss_accuracy'] = accuracy_score(y_test, lr.predict(X_test))

lr = LogisticRegression()
lr.fit(X_train, y_train)
out['sklearn_accuracy'] = accuracy_score(y_test, lr.predict(X_test))

out['mse_error_first'] = mse_err[0]
out['mse_error_last'] = mse_err[1]
out['logloss_error_first'] = logloss_err[0]
out['logloss_error_last'] = logloss_err[1]

print(out)

print('''Answers to the questions:
1) 0.00000
2) 0.00000
3) 0.00153
4) 0.006
5) expanded
6) expanded''')
