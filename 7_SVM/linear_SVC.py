# @Time    : 2018/12/23 14:08
# @Author  : Xu Huipeng
# @Blog    : https://brycexxx.github.io/

import numpy as np
from copy import deepcopy
from typing import Tuple
from sklearn.preprocessing import LabelBinarizer

class SupportVectorClassifier:
    """
    线性支持向量机
    """
    def __init__(self, tol: float=1e-4, C: float=1.0, max_iter: int=1000):
        self.tol = tol
        self.C = C
        self.max_iter = max_iter

    def _gx(self, x: np.ndarray) -> np.ndarray:
        inner_product = np.dot(self.X, x)
        gx = np.dot(np.multiply(inner_product, self.y).T, self.alpha) + self._b
        return gx

    def _select_alpha1(self) -> int:
        kkt_condition = np.zeros((self._n_samples, 1))
        for i in range(self._n_samples):
            gx = self._Ei[i] + self.y[i]
            ygx = gx * self.y[i]
            alpha_i = self.alpha[i]
            if ygx == 1 and (alpha_i >= self.C or alpha_i <= 0):
                kkt_condition[i] += 2
            if ygx >= 1 and alpha_i != 0:
                kkt_condition[i] += 1
            if ygx <= 1 and alpha_i != self.C:
                kkt_condition[i] += 1
        idx = kkt_condition.argmax()
        return idx

    def _get_L_H(self, y1: int, y2: int, alpha1: float, alpha2: float) -> Tuple[float, float]:
        if y1 == y2:
            L = max(.0, alpha2 + alpha1 - self.C)
            H = min(self.C, alpha2 + alpha1)
        else:
            L = max(.0, alpha2 - alpha1)
            H = min(self.C, self.C + alpha2 - alpha1)
        return L, H

    def _stop_condition(self) -> bool:
        cond1 = np.dot(self.y.T, self.alpha) == 0
        if not cond1:
            return False
        cond2 = (self.alpha >= 0).all() and (self.alpha <= self.C).all()
        if not cond2:
            return False
        for i in range(self._n_samples):
            gx = self._Ei[i] + self.y[i]
            ygx = gx * self.y[i]
            if self.alpha[i] == 0:
                if ygx < 1:
                    return False
            elif 0 < self.alpha[i] < self.C:
                if ygx != 1:
                    return False
            else:
                if ygx > 1:
                    return False
        return True

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.X, self.y = X, y
        self._b = 0
        lb = LabelBinarizer(neg_label=-1)
        self.y = lb.fit_transform(self.y)
        # self.classes = lb.classes_
        self._n_samples, self._n_features = self.X.shape
        self.alpha = np.zeros((self._n_samples, 1))
        self._Ei = self._gx(self.X.T) - self.y
        for _ in range(self.max_iter):
            idx1 = self._select_alpha1()
            alpha1 = self.alpha[idx1]
            E1 = self._Ei[idx1]
            idx2 = self._Ei.argmin() if E1 >= 0 else self._Ei.argmax()
            E2 = self._Ei[idx2]
            alpha2 = self.alpha[idx2]
            K_11 = np.dot(self.X[idx1, np.newaxis], self.X[idx1][:, np.newaxis])
            K_22 = np.dot(self.X[idx2, np.newaxis], self.X[idx2][:, np.newaxis])
            K_12 = np.dot(self.X[idx1, np.newaxis], self.X[idx2][:, np.newaxis])
            eta =  K_11 + K_22 - 2 * K_12
            alpha2_new_unc = alpha2 + self.y[idx2] * (E1 - E2) / eta
            L, H = self._get_L_H(self.y[idx1], self.y[idx2], alpha1, alpha2)
            # 更新后的 alpha2
            if alpha2_new_unc > H:
                alpha2_new = H
            elif alpha2_new_unc < L:
                alpha2_new = L
            else:
                alpha2_new = alpha2_new_unc
            # 更新后的 alpha1
            alpha1_new = alpha1 + self.y[idx1] * self.y[idx2] * (alpha2 - alpha2_new)
            # 更新 alpha1 alpha2
            last_alpha = deepcopy(self.alpha)
            self.alpha[[idx1, idx2]] = np.array([[alpha1_new, alpha2_new]]).T
            # 若在精度范围 ε 内满足停止条件则跳出循环
            if np.linalg.norm(self.alpha - last_alpha) < self.tol and self._stop_condition():
                break
            b1_new = - E1 - self.y[idx1] * K_11 * (alpha1_new - alpha1) - self.y[idx2] * \
                     K_12 * (alpha2_new - alpha2) + self._b
            b2_new = - E2 - self.y[idx1] * K_12 * (alpha1_new - alpha1) - self.y[idx2] * \
                     K_22 * (alpha2_new - alpha2) + self._b
            # 更新 self._b
            if 0 < alpha1_new < self.C:
                self._b = b1_new
            elif 0 < alpha2_new < self.C:
                self._b = b2_new
            else:
                self._b = (b1_new + b2_new) / 2
            # 更新 Ei
            self._Ei = self._gx(self.X.T) - self.y
        return self

    def predict(self, X: np.ndarray):
        gx = self._gx(X.T)
        ret = np.where(gx>=0, 1, 0)
        return ret.squeeze()

if __name__ == "__main__":
    from sklearn.datasets import load_breast_cancer
    from sklearn.svm import LinearSVC
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    from sklearn.dummy import DummyClassifier

    bc = load_breast_cancer()
    X = bc.data
    y = bc.target
    train_X, test_X, train_y, test_y = train_test_split(X, y, random_state=0)
    svc = SupportVectorClassifier(max_iter=3000, tol=1e-5)
    svc.fit(train_X, train_y)
    y_pred = svc.predict(test_X)
    print(y_pred.sum())
    print(accuracy_score(test_y, y_pred))
    print('-'*20)
    sk_svc = LinearSVC(max_iter=2000, random_state=42).fit(train_X, train_y)
    print(sk_svc.score(test_X, test_y))






