# @Time    : 2019/1/10 16:08
# @Author  : Xu Huipeng
# @Blog    : https://brycexxx.github.io/
import numpy as np
from typing import Optional
from warnings import warn


class HiddenMarkovModel:
    """
    隐马尔科夫模型
    """

    def __init__(self, n_states: int = 1, eps: float = 1e-3, max_iter: int = 100, random_state: Optional[int] = None,
                 start_prob: Optional[np.ndarray] = None,
                 transmat_prob: Optional[np.ndarray] = None,
                 obs_prob: Optional[np.ndarray] = None):
        self.n_states = n_states
        self.start_prob = start_prob
        self.transmat_prob = transmat_prob
        self.obs_prob = obs_prob
        self.eps = eps
        self.max_iter = max_iter
        self.random_state = random_state

    def simulate(self, T: int):
        """
        生成观测序列
        借鉴于 http://www.hankcs.com/ml/hidden-markov-model.html
        """

        def draw_from(probs):
            return np.where(np.random.multinomial(1, probs) == 1)[0][0]

        observations = np.zeros(T, dtype=int)
        states = np.zeros(T, dtype=int)
        states[0] = draw_from(self.start_prob)
        observations[0] = draw_from(self.obs_prob[states[0], :])
        for t in range(1, T):
            states[t] = draw_from(self.transmat_prob[states[t - 1], :])
            observations[t] = draw_from(self.obs_prob[states[t], :])
        return observations, states

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
        beta = np.ones((length, self.n_states))
        for i in range(length - 2, -1, -1):
            beta[i, :] = np.dot(self.transmat_prob, self.obs_prob[:, obs_seq[i + 1]] * beta[i + 1, :]).T
        return beta

    def backward_prob(self, obs_seq: np.ndarray):
        beta = self._backward_prob_dist(obs_seq)
        bp = np.dot(self.start_prob, self.obs_prob[:, obs_seq[0]].reshape(-1, 1) * beta[0:1, :].T)[0]
        return bp

    def fit(self, X: np.ndarray):
        n_samples, length = X.shape
        obs_set = np.unique(X)
        n_obs = np.size(obs_set)
        # 随机初始化参数,保证每行相加等于 1
        rs = np.random.RandomState(self.random_state)
        transmat_prob = rs.random_sample((self.n_states, self.n_states))
        self.transmat_prob = transmat_prob / transmat_prob.sum(axis=1, keepdims=True)
        obs_prob = rs.random_sample((self.n_states, n_obs))
        self.obs_prob = obs_prob / obs_prob.sum(axis=1, keepdims=True)
        start_prob = rs.random_sample((self.n_states,))
        self.start_prob = start_prob / start_prob.sum()
        for i in range(self.max_iter):
            print('iter %d times' % i)
            xi = np.zeros((self.n_states, self.n_states))
            gamma = np.zeros((length, self.n_states))
            obs_numerator = np.zeros((self.n_states, n_obs))
            for x in X:
                alpha = self._forward_prob_dist(x)
                beta = self._backward_prob_dist(x)
                xi_t = np.zeros((self.n_states, self.n_states))
                for t in range(length - 1):
                    xi_numerator = alpha[t].reshape(-1, 1) * self.transmat_prob * \
                                   self.obs_prob[:, x[t + 1]] * beta[t + 1]
                    xi_t += xi_numerator / (xi_numerator.sum())
                xi += xi_t
                gamma_numerator = alpha * beta
                gamma_i = gamma_numerator / (gamma_numerator.sum(axis=1, keepdims=True))
                gamma += gamma_i
                for k in range(n_obs):
                    idx = x == obs_set[k]
                    obs_numerator[:, k] += gamma_i[idx].sum(axis=0)
            old_start = self.start_prob.copy()
            old_transmat = self.transmat_prob.copy()
            old_obs = self.obs_prob.copy()
            self.start_prob = gamma[0] / n_samples
            self.transmat_prob = xi / (gamma[:length - 1].sum(axis=0, keepdims=True).T)
            self.obs_prob = obs_numerator / (gamma.sum(axis=0).reshape(-1, 1))
            if np.linalg.norm(self.start_prob - old_start) < self.eps and \
                    np.linalg.norm(self.transmat_prob - old_transmat) < self.eps and \
                    np.linalg.norm(self.obs_prob - old_obs) < self.eps: return self
        warn("Maybe hmm doesn't converge!")
        return self

    # 使用 viterbi 算法预测
    def predict(self, X: np.ndarray):
        n_samples, length = X.shape
        state_seq = np.zeros((n_samples, length), dtype=np.int16)
        for i in range(n_samples):
            delta = np.zeros((length, self.n_states))
            delta[0] = self.start_prob * self.obs_prob[:, X[i, 0]]
            psi = np.zeros((length, self.n_states))
            for t in range(1, length):
                temp_prob = delta[t - 1].reshape(-1, 1) * self.transmat_prob
                delta[t] = temp_prob.max(axis=0) * self.obs_prob[:, X[i, t]]
                psi[t] = temp_prob.argmax(axis=0)
            state_seq[i, -1] = delta[-1].argmax()
            # 最优路径回溯
            for t in range(length - 2, -1, -1):
                state_seq[i, t] = psi[t, state_seq[i, t + 1]]
        return state_seq


if __name__ == "__main__":
    # hmm = HiddenMarkovModel()
    # hmm.start_prob = np.array([0.2, 0.4, 0.4])
    # hmm.transmat_prob = np.array([[0.5, 0.2, 0.3], [0.3, 0.5, 0.2], [0.2, 0.3, 0.5]])
    # hmm.obs_prob = np.array([[0.5, 0.5], [0.4, 0.6], [0.7, 0.3]])
    from hmmlearn.hmm import MultinomialHMM
    from sklearn.metrics import accuracy_score
    import numpy as np

    np.set_printoptions(precision=5, suppress=True)

    # h = MultinomialHMM(n_components=3)
    # h.startprob_ = np.array([0.2, 0.4, 0.4])
    # h.transmat_ = np.array([[0.5, 0.2, 0.3], [0.3, 0.5, 0.2], [0.2, 0.3, 0.5]])
    # h.emissionprob_ = np.array([[0.5, 0.5], [0.4, 0.6], [0.7, 0.3]])
    # X = h.sample(n_samples=1200, random_state=0)[0]
    hmm_generate = HiddenMarkovModel(n_states=3)
    hmm_generate.start_prob = np.array([0.2, 0.4, 0.4])
    hmm_generate.transmat_prob = np.array([[0.5, 0.2, 0.3], [0.3, 0.5, 0.2], [0.2, 0.3, 0.5]])
    hmm_generate.obs_prob = np.array([[0.5, 0.5], [0.4, 0.6], [0.7, 0.3]])
    X, I = hmm_generate.simulate(100)
    # hmm = HiddenMarkovModel(n_states=3, eps=0.01, max_iter=500)
    # hmm.fit(X.reshape(1, -1))
    # print('初始状态分布概率：')
    # print(hmm.start_prob)
    # print('状态转移概率矩阵：')
    # print(hmm.transmat_prob)
    # print(hmm.transmat_prob.sum(axis=1, keepdims=True))
    # print('观测概率矩阵：')
    # print(hmm.obs_prob)
    # print(hmm.obs_prob.sum(axis=1))
    # I_pred = hmm.predict(X.reshape(1, -1))
    # print(accuracy_score(I, I_pred.squeeze()))
    hmmL = MultinomialHMM(n_components=3).fit(X.reshape(-1, 1))
    print('from hmmlearn:')
    print(hmmL.startprob_)
    print(hmmL.transmat_)
    print(hmmL.emissionprob_)
    print(accuracy_score(I, hmmL.predict(X.reshape(-1, 1))))
