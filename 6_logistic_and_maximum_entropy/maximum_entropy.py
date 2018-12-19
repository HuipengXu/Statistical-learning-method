# @Time    : 2018/12/14 9:50
# @Author  : Xu Huipeng
# @Blog    : https://brycexxx.github.io/

import numpy as np
from sklearn.datasets import make_spd_matrix
from sklearn.preprocessing import LabelBinarizer
from copy import deepcopy

class MaximumEntropyIIS:

    def __init__(self, max_iter: int=200, eps: float=1e-2):
        self._max_iter = max_iter
        self._eps = eps

    def _convergence(self, last_w):
        for w, lw in zip(self._w.flatten().tolist(), last_w.flatten().tolist()):
            if abs(w - lw) >= self._eps:
                return False
        return True

    def _calculate_pyx(self, X: np.ndarray):
        numerator = np.exp(np.dot(X, self._w.T))
        denominator = numerator.sum(axis=1, keepdims=True)
        return numerator / denominator

    def fit(self, X: np.ndarray, y: np.ndarray):
        n_samples = X.shape[0]
        lb = LabelBinarizer()
        y = lb.fit_transform(y)
        self.classes = lb.classes_
        y = np.concatenate((1-y, y), axis=1)
        # 特征 (x, y) 出现的频数, 实际上是统计了每个特征与 y 配对之后的频数，将 X = (x1, x2, x3, ...), y = 0, 拆为 (x1, 0), (x2, 0), (x3, 0), ...
        xy_freq = np.dot(y.T, X)
        # 所有特征在 (x, y) 出现的次数
        M = xy_freq.sum()
        # 特征函数 f(x, y) 关于经验分布的 P(x, y) 的期望值
        self._E_P_wave = xy_freq / n_samples
        # 待优化参数
        self._w = np.zeros_like(self._E_P_wave)
        last_w = deepcopy(self._w)
        for _ in range(self._max_iter):
            pyx = self._calculate_pyx(X)
            E_P = np.dot(pyx.T, X / n_samples)
            delta = 1.0 / M * (np.log(self._E_P_wave + 1e-9) - np.log(E_P + 1e-9))
            self._w += delta
            if self._convergence(last_w):
                break
            last_w = deepcopy(self._w)
        return self

    def predict_proba(self, X: np.ndarray):
        return self._calculate_pyx(X)

    def predict(self, X: np.ndarray):
        proba = self.predict_proba(X)
        idx = proba.argmax(axis=1)
        ret = self.classes[idx]
        return ret

class MaximumEntropyBFGS:

    def __init__(self, max_iter: int=200, eps: float=1e-2):
        self._max_iter = max_iter
        self._eps = eps

    def _calculate_pyx(self, X: np.ndarray):
        w = np.concatenate((self._w[:self.features], self._w[self.features:]), axis=1)
        numerator = np.exp(np.dot(X, w))
        denominator = numerator.sum(axis=1, keepdims=True)
        return numerator / denominator

    def _cost(self, X:np.ndarray, w: np.ndarray):
        w = np.concatenate((w[:self.features], w[self.features:]), axis=1)
        P_x = X / X.shape[0]
        log_zw = np.log(np.exp(np.dot(X, w)).sum(axis=1, keepdims=True))
        cost = np.multiply(P_x.prod(axis=1, keepdims=True), log_zw).sum() - np.multiply(self._E_P_wave, np.concatenate((w[:, 0], w[:, 1]), axis=0)).sum()
        return cost

    def _calculate_gradient(self, X: np.ndarray):
        pyx = self._calculate_pyx(X)
        E_P = np.dot(pyx.T, X / X.shape[0]).reshape(-1, 1)
        g_w = E_P - self._E_P_wave
        return g_w

    # 确定步长 lambda_
    # Armijo 条件: f(xk + lambda_ * pk) <= f(xk) + lambda_ * c * pk.T * g
    def _backtracking_line_search(self, X:np.ndarray, g_w: np.ndarray, p: np.ndarray, c, tau):
        lambda_ = 10.0
        while True:
            new_w = self._w + lambda_ * p
            # Armijo 条件左边
            Armijo_l = self._cost(X, new_w)
            # Armijo 条件右边
            Armijo_r = self._cost(X, self._w) + lambda_ * c * np.dot(p.T, g_w)
            if Armijo_l <= Armijo_r:
                return lambda_
            lambda_ *= tau

    def fit(self, X: np.ndarray, y: np.ndarray):
        # 初始化 c, tau
        c = np.random.random_sample()
        tau = np.random.random_sample()
        n_samples, self.features = X.shape
        lb = LabelBinarizer().fit(y)
        self.classes = lb.classes_
        y = lb.transform(y)
        y = np.concatenate((1-y, y), axis=1)
        xy_freq = np.dot(y.T, X).reshape(-1, 1)
        self._E_P_wave = xy_freq / n_samples
        self._w = np.zeros_like(self._E_P_wave)
        # 初始化对称正定矩阵 B
        # B_inv = np.eye(self._w.shape[0])
        I = np.eye(self._w.shape[0])
        # B = make_spd_matrix(self._w.shape[0], random_state=0)
        B_inv = I
        g_w = self._calculate_gradient(X)
        for _ in range(self._max_iter):
            p = - np.dot(B_inv, g_w)
            lambda_ = self._backtracking_line_search(X, g_w, p, c, tau)
            last_w = deepcopy(self._w)
            self._w += lambda_ * p
            last_g_w = deepcopy(g_w)
            g_w = self._calculate_gradient(X)
            if np.linalg.norm(g_w) < self._eps:
                break
            yk= g_w - last_g_w
            sk = self._w - last_w
            # B = B + np.dot(yk, yk.T) / (np.dot(yk.T, s) + 1e-9) -\
            #     np.dot(np.dot(np.dot(B, s), s.T), B) / (np.dot(np.dot(s.T, B), s) + 1e-9)
            rho = 1 / (np.dot(yk.T, sk) + 1e-20)
            B_l = (I - rho * np.dot(sk, yk.T))
            B_r = (I - rho * np.dot(yk, sk.T))
            B_inv = np.dot(B_l, np.dot(B_inv, B_r)) + rho * np.dot(sk, sk.T)
        return self

    def predict_proba(self, X: np.ndarray):
        return self._calculate_pyx(X)

    def predict(self, X: np.ndarray):
        proba = self.predict_proba(X)
        idx = proba.argmax(axis=1)
        ret = self.classes[idx]
        return ret

if __name__ == "__main__":
    import pandas as pd
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.preprocessing import binarize

    with open(r'..\ttest\data.txt', 'r') as f:
        X = []
        y = []
        for line in f.readlines():
            l = line.split()
            X.append(' '.join(l[1:]))
            y.append(l[0])
    y = np.array(y)
    y = np.where(y=='yes', 1, 0)
    vect = CountVectorizer()
    vectorized_X = vect.fit_transform(X)
    me = MaximumEntropyIIS().fit(vectorized_X.toarray(), y)
    x_test = ['sunny hot high FALSE', 'overcast hot high FALSE', 'sunny cool high TRUE']
    x_test_vect = vect.transform(x_test)
    pred = me.predict(vectorized_X.toarray())
    print(accuracy_score(y, pred))
    # pred = me.predict(x_test_vect.toarray())
    # print(pred)

    # data = open(r'..\4_naive_bayes\corpus').read()
    # labels, texts = [], []
    # for line in data.split('\n'):
    #     content = line.split(maxsplit=1)
    #     labels.append(content[0][-1])
    #     texts.append(content[1])
    #
    # data = pd.DataFrame(data={'labels': labels, 'texts': texts})
    # X_train, X_test, y_train, y_test = train_test_split(data['texts'], data['labels'].apply(int), random_state=0)
    # vect = CountVectorizer(min_df=5).fit(X_train)
    # X_train_vectorized = vect.transform(X_train).toarray()
    # X_train_vectorized = binarize(X_train_vectorized)
    # X_test_vectorized = vect.transform(X_test).toarray()
    # X_test_vectorized = binarize(X_test_vectorized)
    # me = MaximumEntropyBFGS(500).fit(X_train_vectorized, y_train)
    # y_pred = me.predict(X_test_vectorized)
    # print(accuracy_score(y_test, y_pred))
