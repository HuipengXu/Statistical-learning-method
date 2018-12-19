# @Time    : 2018/12/12 19:30
# @Author  : Xu Huipeng
# @Blog    : https://brycexxx.github.io/

import numpy as np

# 逻辑回归分类
class LogisticRegression:

    def __init__(self, learning_rate: float=1e-3, decay: float=1e-4, C: float=1.0, max_iter: int=5000, solver: str='sgd'):
        self._w = None
        self._lr = learning_rate
        self._decay = decay
        self._C = C
        self._max_iter = max_iter
        self._solver = solver

    def _sigmoid(self, z: np.ndarray):
        return 1.0 / (np.exp(-z) + 1)

    def _calculate_grad(self, x: np.ndarray, y: np.ndarray, w: np.ndarray):
        z = np.dot(x, w)
        grad = np.dot(x.T, (y - self._sigmoid(z))) + w / self._C
        return grad
    
    def _gradient_descent(self, X: np.ndarray, y: np.ndarray, n_features: int):
        w = np.zeros((n_features, 1))
        for i in range(self._max_iter):
            lr = self._lr * 1.0 / (1.0 + self._decay * i)
            if self._solver == 'sgd':
                idx = np.random.randint(X.shape[0])
                x, y_ = X[idx: idx+1, :], y[idx]
            else:
                x, y_ = X, y
            grad = self._calculate_grad(x, y_.reshape(-1, 1), w)
            w += lr * grad
        return w

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        :param X: train data, X.shape = (n_samples, n_features)
        :param y: label, y.shape = (n_samples, )
        :return: self
        """
        X = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)
        rows, features = X.shape
        self._classes = np.unique(y)
        self._classes_count = np.size(self._classes)
        # 二项逻辑回归
        if self._classes_count == 2:
            self._w = self._gradient_descent(X, y, features)
            return self
        # 多项逻辑回归
        self._w = np.zeros((features, self._classes_count))
        for i in range(self._classes_count):
            y_c = np.where(y==self._classes[i], 1, 0)
            self._w[:, i] = self._gradient_descent(X, y_c, features).squeeze()
        return self

    def predict(self, X: np.ndarray):
        X = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)
        proba = self._sigmoid(np.dot(X, self._w))
        if self._classes_count == 2:
            y_pred = np.where(proba >= 0.5, 1, 0)
        else:
            ret_index = proba.argmax(axis=1)
            y_pred = self._classes[ret_index]
        return y_pred


if __name__ == "__main__":
    from sklearn.datasets import load_breast_cancer, load_iris, load_digits
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression as LR
    import matplotlib.pyplot as plt
    import pandas as pd

    # data = pd.read_csv('data.txt', sep='\s+', header=None)
    # X_train = data.iloc[:, :2]
    # y_train = data.iloc[:, 2]
    # lr = LogisticRegression(C=0.08).fit(X_train, y_train)
    # neg_index = (data.iloc[:, 2] == 0).tolist()
    # pos_index = (data.iloc[:, 2] == 1).tolist()
    # plt.scatter(data.iloc[pos_index, 0], data.iloc[pos_index, 1], s=30, c='red')
    # plt.scatter(data.iloc[neg_index, 0], data.iloc[neg_index, 1], s=30, c='green')
    # x = np.arange(-3, 3, 0.1)
    # y = (- lr._w[2] * 1 - lr._w[0] * x) / lr._w[1]
    # # y = (lr.intercept_[0] * 1 - lr.coef_[0, 1] * x) / lr.coef_[0, 0]
    # plt.plot(x, y)
    # plt.show()
    bc = load_digits()
    X = bc.data
    y = bc.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    lr = LogisticRegression(C=5).fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    print(accuracy_score(y_test, y_pred))
    print('------------------')
    Lr = LR(solver='sag', max_iter=1000).fit(X_train, y_train)
    print(Lr.score(X_test, y_test))
