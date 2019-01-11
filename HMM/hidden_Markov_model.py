# @Time    : 2019/1/10 16:08
# @Author  : Xu Huipeng
# @Blog    : https://brycexxx.github.io/
import numpy as np
from typing import Optional


class HiddenMarkovModel:
    """
    隐马尔科夫模型
    """

    def __init__(self, n_states: int = 1, start_prob: Optional[np.ndarray] = None,
                 transmat_prob: Optional[np.ndarray] = None,
                 obs_prob: Optional[np.ndarray] = None):
        self.n_states = n_states
        self.start_prob = start_prob
        self.transmat_prob = transmat_prob
        self.obs_prob = obs_prob

    def _forward_prob_dist(self, obs_seq: np.ndarray):
        length = np.size(obs_seq)
        self.n_states = self.start_prob.shape[0]
        alpha = np.zeros((length, self.n_states))
        alpha[0, :] = self.start_prob * self.obs_prob[:, obs_seq[0]]
        for i in range(1, length):
            alpha[i, :] = np.dot(alpha[i - 1, :], self.transmat_prob) * self.obs_prob[:, obs_seq[i]]
        return alpha

    def forward_prob(self, obs_seq: np.ndarray):
        alpha = self._forward_prob_dist(obs_seq)
        return alpha[-1, :].sum()

    def _backward_prob_dist(self, obs_seq: np.ndarray):
        length = np.size(obs_seq)
        self.n_states = self.start_prob.shape[0]
        beta = np.ones((length + 1, self.n_states))
        for i in range(length - 1, 0, -1):
            beta[i, :] = np.dot(self.transmat_prob, self.obs_prob[:, obs_seq[i]] * beta[i + 1, :]).T
        beta[0, :] = self.start_prob * self.obs_prob[:, obs_seq[0]] * beta[1, :]
        return beta[:length]

    def backward_prob(self, obs_seq: np.ndarray):
        beta = self._backward_prob_dist(obs_seq)
        return beta[0, :].sum()

    def fit(self, X: np.ndarray):
        obs_set = np.unique(X)
        self.transmat_prob = np.ones((self.n_states, self.n_states)) / self.n_states
        self.obs_prob = np.ones((self.n_states, np.size(obs_set))) / np.size(obs_set)
        self.start_prob = np.ones((self.n_states,)) / self.n_states
        while True:
            pass


if __name__ == "__main__":
    hmm = HiddenMarkovModel()
    hmm.start_prob = np.array([0.2, 0.4, 0.4])
    hmm.transmat_prob = np.array([[0.5, 0.2, 0.3], [0.3, 0.5, 0.2], [0.2, 0.3, 0.5]])
    hmm.obs_prob = np.array([[0.5, 0.5], [0.4, 0.6], [0.7, 0.3]])
    obs_seq = np.array([0, 1, 0])
    print(hmm.forward_prob(obs_seq))
    print(hmm.backward_prob(obs_seq))
