import numpy as np 
from scipy.misc import logsumexp
from collections import Counter
from BaseNB import BaseNaiveBayes

class MultinomialNaiveBayes(BaseNaiveBayes):

    def __init__(self, alpha: float=1.0):
        """
        alpha: 平滑参数, 防止条件概率分母为 0 的情况
        """
        super(MultinomialNaiveBayes, self).__init__()
        self._alpha = alpha
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        X: train dataset, shape = (n_samples, n_features)
        y: target, shape = (n_samples, )
        """
        # 计算 y 的先验概率
        y_prior_proba = []
        # features = X.shape[1]
        self.classes_ = np.unique(y)
        for c in self.classes_:
            c_count = (y==c).sum()
            y_prior_proba.append(c_count / np.size(y))
        # y 的先验概率
        self._y_prior_proba = np.array(y_prior_proba)
        # # 特征 Xi 在所属类别 yk 中出现的总次数
        # self._N_yk_xi = []
        # 所有特征在 yk 中出现的总次数
        self._N_yk = np.zeros((np.size(self.classes_), 1))
        self._N_yk_xj = []
        for i in range(np.size(self.classes_)):
            x_given_y = X[y==self.classes_[i]]
            self._N_yk_xj.append([Counter(x) for x in x_given_y.T])
            self._N_yk[i, :] = x_given_y.shape[0]
        return self

    def _joint_log_likelihood(self, X: np.ndarray):
        rows, features = X.shape
        classes_nums = np.size(self.classes_)
        log_proba = np.zeros((rows, classes_nums))
        for m in range(rows): 
            for i in range(classes_nums):
                x_given_y_prior = np.zeros((1, features))
                for j in range(features):
                    numerator = self._N_yk_xj[i][j].get(X[m, j], 0) + self._alpha
                    denominator = self._N_yk[i] + self._alpha * features
                    x_given_y_prior[:, j] = numerator / denominator
                log_x_given_y_prior = np.sum(np.log(x_given_y_prior)) + np.log(self._y_prior_proba[i])
                log_proba[m, i] = log_x_given_y_prior
        return log_proba

    def __repr__(self):
        return "<MultinomialNaiveBayes>" 

if __name__ == "__main__":
    from sklearn.datasets import load_iris
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split
    from sklearn.naive_bayes import MultinomialNB

    X = np.array([[1, 'S'], [1, 'M'], [1, 'M'], [1, 'S'], [1, 'S'], [2, 'S'], [2, 'M'], 
                    [2, 'M'], [2, 'L'], [2, 'L'], [3, 'L'], [3, 'M'], [3, 'M'], [3, 'L'], [3, 'L']])
    y = np.array([-1, -1, 1, 1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, -1])
    # iris = load_iris()
    # X = iris.data 
    # y = iris.target
    # X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    mnb = MultinomialNaiveBayes(0).fit(X, y)
    y_pred = mnb.predict(X)
    print(accuracy_score(y.T, y_pred))
    # print(X[1, :])