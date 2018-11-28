import numpy as np
from collections import Counter

# 朴素贝叶斯
class NaiveBayesClassifier:

    def __init__(self, algorithm="Gaussian"):
        self._algorithm = algorithm
        self._x_priori_distribution = None
        self._y_priori_distribution = None

    # 训练
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        X: train dataset, shape = (n_samples, n_features)
        """
        if self._algorithm == "Gaussian":
            self._x_priori_distribution, self._y_priori_distribution = self._gaussian(X, y)
            return self
        elif self._bernoulli == "bernoulli":
            pass
        else:
            pass

    # 高斯模型
    def _gaussian(self, X: np.ndarray, y: np.ndarray):
        # 计算 y 的先验概率
        y_counter = Counter(y)
        samples = y.size
        y_priori_probability = {c: freq / samples for c, freq in y_counter.items()}
        # 利用高斯分布计算属于特定类别 y 的连续变量 x 的先验分布
        _, features = X.shape
        catagories = y_counter.keys()
        x_c_distribution_dict = {}
        for c in catagories:
            # 获取类别为 c 的 X
            x_c = X[y==c]
            for fea in range(features):
                mean = x_c[fea].mean()
                std = x_c[fea].std()
                x_c_distribution = lambda x: 1 / np.sqrt(2 * np.pi * std) * np.exp(- (x - mean) ** 2 / (2 * std))
                x_c_distribution_dict[(fea, c)] = x_c_distribution
        return x_c_distribution_dict, y_priori_probability

    # 伯努利模型
    def _bernoulli(self, X: np.ndarray, y: np.ndarray):
        pass

    # 多项式模型
    def _multinomizl(self, X: np.ndarray, y: np.ndarray):
        pass

    # 预测
    def predict(self, X: np.ndarray):
        """
        X: predict, shape = (n_samples, n_features)
        """
        y_predict_proba = self.predict_proba(X)
        return y_predict_proba.argmax(axis=1)

    # 置信度计算
    def predict_proba(self, X: np.ndarray):
        _, features = X.shape
        y_pred = []
        for x in X:
            x_pred = []
            for c in self._y_priori_distribution.keys():
                x_c_proba = self._y_priori_distribution[c]
                for fea in range(features):
                    x_c_fea_proba = self._x_priori_distribution[(fea, c)](x[fea])
                    x_c_proba *= x_c_fea_proba
                x_pred.append(x_c_proba)
            y_pred.append(x_pred)
        return np.array(y_pred)

    def __repr__(self):
        return "<NaiveBayesClassifier: %s>" % self._algorithm

if __name__ == "__main__":
    from sklearn.datasets import load_iris
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split

    iris = load_iris()
    X = iris.data
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    nbc = NaiveBayesClassifier().fit(X_train, y_train)
    y_pred = nbc.predict(X_test)
    print(accuracy_score(y_test, y_pred))