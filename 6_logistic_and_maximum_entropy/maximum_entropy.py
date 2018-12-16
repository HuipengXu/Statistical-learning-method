# @Time    : 2018/12/14 9:50
# @Author  : Xu Huipeng
# @Blog    : https://brycexxx.github.io/

import numpy as np
from sklearn.preprocessing import LabelBinarizer
from copy import deepcopy

class MaximumEntropy:

    def __init__(self, max_iter=200, tol=1e-2):
        self._max_iter = max_iter
        self._tol = tol

    def _convergence(self, last_w):
        for w, lw in zip(self._w.flatten().tolist(), last_w.flatten().tolist()):
            if abs(w - lw) >= self._tol:
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
        E_P_wave = xy_freq / n_samples
        # 待优化参数
        self._w = np.zeros_like(E_P_wave)
        last_w = deepcopy(self._w)
        for _ in range(self._max_iter):
            pyx = self._calculate_pyx(X)
            E_P = np.dot(pyx.T, X / n_samples)
            delta = 1.0 / M * (np.log(E_P_wave + 1e-9) - np.log(E_P + 1e-9))
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
    me = MaximumEntropy().fit(vectorized_X.toarray(), y)
    x_test = ['sunny hot high FALSE', 'overcast hot high FALSE', 'sunny cool high TRUE']
    x_test_vect = vect.transform(x_test)
    pred = me.predict(vectorized_X.toarray())
    print(accuracy_score(y, pred))

    # data = open(r'..\4_naive_bayes\corpus').read()
    # labels, texts = [], []
    # for line in data.split('\n'):
    #     content = line.split(maxsplit=1)
    #     labels.append(content[0][-1])
    #     texts.append(content[1])
    #
    # data = pd.DataFrame(data={'labels': labels, 'texts': texts})
    # X_train, X_test, y_train, y_test = train_test_split(data['texts'], data['labels'].apply(int), random_state=0)
    # vect = CountVectorizer(ngram_range=(1, 3), min_df=3).fit(X_train)
    # X_train_vectorized = vect.transform(X_train).toarray()
    # X_train_vectorized = binarize(X_train_vectorized)
    # X_test_vectorized = vect.transform(X_test).toarray()
    # X_test_vectorized = binarize(X_test_vectorized)
    # me = MaximumEntropy().fit(X_train_vectorized, y_train)
    # y_pred = me.predict(X_test_vectorized)
    # print(accuracy_score(y_test, y_pred))
