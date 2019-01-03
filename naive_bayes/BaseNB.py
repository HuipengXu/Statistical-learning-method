import numpy as np
from scipy.misc import logsumexp

class BaseNaiveBayes:

    def __init__(self):
        self.classes_ = None

    def _joint_log_likelihood(self, X: np.ndarray):
        """
        x 的非归一化的后验概率的计算
        """
        pass

    def predict_log_proba(self, X: np.ndarray):
        jll = self._joint_log_likelihood(X)
        # 归一化
        log_prob_x = logsumexp(jll, axis=1)
        return jll - np.atleast_2d(log_prob_x).T
         
    def predict_proba(self, X: np.ndarray):
        return np.exp(self.predict_log_proba(X))

    def predict(self, X: np.ndarray):
        jll = self._joint_log_likelihood(X)
        return self.classes_[np.argmax(jll, axis=1)]