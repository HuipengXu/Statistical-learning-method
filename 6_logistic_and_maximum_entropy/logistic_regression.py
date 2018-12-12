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

    def _calculate_grad(self, X: np.ndarray, y: np.ndarray):
        minuend = np.dot(X.T, y.reshape(-1, 1))
        subtrahend = np.dot(X.T, np.exp(np.dot(X, self._w)) / (1+np.exp(np.dot(X, self._w))))
        grad = minuend - subtrahend
        return grad

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        :param X:
        :param y:
        :return:
        """
        X = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)
        rows, features = X.shape
        self._w = np.zeros((features, 1))
        for _ in range(3000):
            grad = self._calculate_grad(X, y)
            # if grad.mean() <= 1e3:
            #     break
            self._w -= self._lr * grad
            print("current accuracy is %.3f" % accuracy_score(y, self.predict(X[:, :-1])))
        return self

    def predict(self, X: np.ndarray):
        X = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)
        exp_w_x = np.exp(np.dot(X, self._w))
        probability_pos = exp_w_x / (1 + exp_w_x)
        probability_neg = 1 / (1 + exp_w_x)
        proba = np.concatenate([probability_neg, probability_pos], axis=1)
        return proba.argmax(axis=1)


if __name__ == "__main__":
    from sklearn.datasets import load_breast_cancer
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split

    bc = load_breast_cancer()
    X = bc.data
    y = bc.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    lr = LogisticRegression().fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    print(y_pred)
    print(accuracy_score(y_test, y_pred))
