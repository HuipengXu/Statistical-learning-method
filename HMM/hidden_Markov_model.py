# @Time    : 2019/1/10 16:08
# @Author  : Xu Huipeng
# @Blog    : https://brycexxx.github.io/
import numpy as np
from typing import Optional


class HiddenMarkovModel:
    """
    隐马尔科夫模型
    """

    def __init__(self, start_prob: Optional[np.ndarray] = None, transmat_prob: Optional[np.ndarray] = None,
                 obs_prob: Optional[np.ndarray] = None):
        self.start_prob = start_prob
        self.transmat_prob = transmat_prob
        self.obs_prob = obs_prob

    def forward(self, obs_seq: np.ndarray):
        length = np.size(obs_seq)
        alpha = self.start_prob * self.obs_prob[:, obs_seq[0]]
        for i in range(1, length):
            alpha = np.dot(alpha, self.transmat_prob) * self.obs_prob[:, obs_seq[i]]
        return alpha.sum()

    def backward(self, obs_seq: np.ndarray):
        length = np.size(obs_seq)
        n_states = self.start_prob.shape[0]
        beta = np.ones((n_states,))
        for i in range(length - 1, 0, -1):
            beta = np.dot(self.transmat_prob, self.obs_prob[:, obs_seq[i]].reshape(-1, 1) * beta.reshape(-1, 1))
        obs_seq_prob = np.dot(self.start_prob.reshape(1, -1),
                              self.obs_prob[:, 0].reshape(-1, 1) * beta.reshape(-1, 1)).squeeze()
        return obs_seq_prob


if __name__ == "__main__":
    hmm = HiddenMarkovModel()
    hmm.start_prob = np.array([0.2, 0.4, 0.4])
    hmm.transmat_prob = np.array([[0.5, 0.2, 0.3], [0.3, 0.5, 0.2], [0.2, 0.3, 0.5]])
    hmm.obs_prob = np.array([[0.5, 0.5], [0.4, 0.6], [0.7, 0.3]])
    obs_seq = np.array([0, 1, 0])
    print(hmm.forward(obs_seq))
    print(hmm.backward(obs_seq))
