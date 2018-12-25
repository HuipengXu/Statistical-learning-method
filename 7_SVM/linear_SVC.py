# @Time    : 2018/12/23 14:08
# @Author  : Xu Huipeng
# @Blog    : https://brycexxx.github.io/

import numpy as np

class SVC:
    """
    线性可分支持向量机
    """
    def __init__(self, tol: float=1e-4, C: float=1.0, max_iter: int=1000):
        self._tol = tol
        self._C = C
        self._max_iter = max_iter
        self._alpha = None

    def _gx(self, X: np.ndarray, y: np.ndarray, idx: int, b):
        inner_product = np.dot(X[idx: idx+1], X.T)
        gx = np.dot(self._alpha.T, np.dot(inner_product, y[:, np.newaxis])) + b
        return gx

    def _select_alpha1(self, X: np.ndarray, y: np.ndarray, b):
        kkt_condition = np.zeros((1, self._n_samples))
        for i in range(self._n_samples):
            gx = self._gx(X, y, i, b)
            ygx = gx * y[i]
            alpha_i = self._alpha[i]
            if ygx == 1 and (alpha_i >= self._C or alpha_i <= 0):
                kkt_condition[i] += 2
            if ygx >= 1 and alpha_i != 0:
                kkt_condition[i] += 1
            if ygx <= 1 and alpha_i != self._C:
                kkt_condition[i] += 1
        idx = kkt_condition.argmax()
        return idx

    def _get_L_H(self, y1: int, y2: int, alpha1: float, alpha2: float):
        if y1 == y2:
            L = max(0, alpha2 + alpha1 - self._C)
            H = min(self._C, alpha2 + alpha1)
        else:
            L = max(0, alpha2 - alpha1)
            H = min(self._C, self._C + alpha2 - alpha1)
        return L, H

    def fit(self, X: np.ndarray, y: np.ndarray):
        self._n_samples, self._n_features = X.shape
        self._alpha = np.zeros((self._n_samples, 1))
        b = 0
        Ei = np.empty([1, self._n_samples])
        for i in range(self._n_samples):
            Ei[i] = self._gx(X, y, i, b) - y[i]
        while True:
            idx1 = self._select_alpha1(X, y, b)
            alpha1 = self._alpha[idx1]
            E1 = self._gx(X, y, idx1, b) - y[idx1]
            idx2 = Ei.argmin() if E1 >= 0 else Ei.argmax()
            E2 = Ei[idx2]
            alpha2 = self._alpha[idx2]
            K_11 = np.dot(X[idx1, np.newaxis], X[np.newaxis, idx1])
            K_22 = np.dot(X[idx2, np.newaxis], X[np.newaxis, idx2])
            K_12 = np.dot(X[idx1, np.newaxis], X[np.newaxis, idx2])
            eta =  K_11 + K_22 - 2 * K_12
            alpha2_new_unc = alpha2 + y[idx2] * (E1 - E2) / eta
            L, H = self._get_L_H(y[idx1], y[idx2], alpha1, alpha2)
            if alpha2_new_unc > H:
                alpha2_new = H
            elif alpha2_new_unc < L:
                alpha2_new = L
            else:
                alpha2_new = alpha2_new_unc
            alpha1_new = alpha1 + y[idx1] * y[idx2] * (alpha2 - alpha2_new)
            flag1 = flag2 = False
            if alpha1_new > 0 and alpha1_new < self._C:
                flag1 = True
                b1_new = -E1 - y[idx1] * K_11 * (alpha1_new - alpha1) - y[idx2] * K_12 * (alpha2_new - alpha2) + b
            if alpha2_new > 0 and alpha2_new < self._C:
                flag2 = True
                b2_new = -E2 - y[idx1] * K_12 * (alpha1_new - alpha1) - y[idx2] * K_22 * (alpha2_new - alpha2) + b





