# @Time    : 2018/12/12 19:30
# @Author  : Xu Huipeng
# @Blog    : https://brycexxx.github.io/

import numpy as np
from sklearn.metrics import accuracy_score

# 二项逻辑回归
class LogisticRegression:

    def __init__(self, learning_rate=1e-3):
        self._w = None
        self._lr = learning_rate

    def sigmoid(self, z: np.ndarray):
        return 1.0 / (np.exp(-z) + 1)

    def _calculate_grad(self, x: np.ndarray, y: np.ndarray):
        z = np.dot(x, self._w)
        grad = np.dot(x.T, (y - self.sigmoid(z)))
        return grad

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        :param X:
        :param y:
        :return:
        """
        X = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)
        self._rows, features = X.shape
        self._w = np.zeros((features, 1))
        it = 0
        for i in range(1000):
            grad = self._calculate_grad(X, y.reshape(-1, 1))
            self._w += self._lr * grad
            it += 1
            print("after iterate %d steps, current accuracy is %.3f" % (i, accuracy_score(y, self.predict(X[:, :-1]))))
        return self

    def predict(self, X: np.ndarray):
        X = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)
        proba = self.sigmoid(np.dot(X, self._w))
        y_pred = np.where(proba >= 0.5, 1, 0)
        return y_pred


if __name__ == "__main__":
    from sklearn.datasets import load_breast_cancer
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split
    import matplotlib.pyplot as plt
    import pandas as pd

    # data = pd.read_csv('data.txt', sep='\s+', header=None)
    # X_train = data.iloc[:, :2]
    # y_train = data.iloc[:, 2]
    # lr = LogisticRegression(0.001).fit(X_train, y_train)
    # neg_index = (data.iloc[:, 2] == 0).tolist()
    # pos_index = (data.iloc[:, 2] == 1).tolist()
    # plt.scatter(data.iloc[pos_index, 0], data.iloc[pos_index, 1], s=30, c='red')
    # plt.scatter(data.iloc[neg_index, 0], data.iloc[neg_index, 1], s=30, c='green')
    # x = np.arange(-3, 3, 0.1)
    # y = (- lr._w[2] * 1 - lr._w[0] * x) / lr._w[1]
    # plt.plot(x, y)
    # plt.show()
    bc = load_breast_cancer()
    X = bc.data
    y = bc.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    lr = LogisticRegression().fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    print(accuracy_score(y_test, y_pred))
