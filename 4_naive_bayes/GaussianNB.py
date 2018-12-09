import numpy as np
from BaseNB import BaseNaiveBayes

# 高斯贝叶斯
class GaussianNaiveBayes(BaseNaiveBayes):

    # 训练
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        X: train dataset, shape = (n_samples, n_features)
        y: target, shape = (n_samples, )
        """
        # 计算 y 的先验概率
        y_prior_proba = []
        self.classes_ = np.unique(y)
        for c in self.classes_:
            c_count = (y == c).sum()
            y_prior_proba.append(c_count / np.size(y))
        self._y_prior_proba = np.array(y_prior_proba)
        # 计算连续变量 x 的高斯分布参数
        features = X.shape[1]
        self._theta = np.zeros((np.size(self.classes_), features))
        self._sigma = np.zeros((np.size(self.classes_), features))
        for i in range(np.size(self.classes_)):
            x_c = X[y == self.classes_[i]]
            self._theta[i, :] = np.mean(x_c, axis=0)
            self._sigma[i, :] = np.var(x_c, axis=0)
        return self

    def _joint_log_likelihood(self, X: np.ndarray):
        jll = []
        for i in range(np.size(self.classes_)):
            log_prior = np.log(self._y_prior_proba[i])
            # 高斯公式取对数
            x_given_y = - 0.5 * np.sum(np.log(2. * np.pi * self._sigma[i, :]))
            x_given_y -= 0.5 * np.sum(((X - self._theta[i, :]) ** 2) / (self._sigma[i, :]), axis=1)
            jll.append(log_prior + x_given_y)
        jll = np.array(jll).T
        return jll
 
    def __repr__(self):
        return "<GaussianNaiveBayes>" 

if __name__ == "__main__":
    from sklearn.datasets import load_iris
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split
    from sklearn.naive_bayes import GaussianNB

    test = np.array([[1, 'S'], [1, 'M'], [1, 'M'], [1, 'S'], [1, 'S'], [2, 'S'], [2, 'M'], 
                    [2, 'M'], [2, 'L'], [2, 'L'], [3, 'L'], [3, 'M'], [3, 'M'], [3, 'L'], [3, 'L']])
    iris = load_iris()
    X = iris.data 
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    gnb = GaussianNaiveBayes().fit(X_train, y_train)
    log_proba = gnb.predict_log_proba(X_test)
    proba = gnb.predict_proba(X_test)
    y_pred = gnb.predict(X_test)
    print(accuracy_score(y_test, y_pred))
