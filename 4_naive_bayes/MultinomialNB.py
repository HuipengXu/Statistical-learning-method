import numpy as np 

class MultinomialNaiveBayes:

    def __init__(self, alpha: float=1.0):
        """
        alpha: 平滑参数, 防止条件概率分母为 0 的情况
        """
        self._alpha = alpha
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        X: train dataset, shape = (n_samples, n_features)
        y: target, shape = (n_samples, )
        """
        # 计算 y 的先验概率
        y_prior_proba = []
        features = X.shape[1]
        self.classes_ = np.unique(y)
        for c in self.classes_:
            c_count = (y==c).sum()
            y_prior_proba.append(c_count / np.size(y))
        # y 的先验概率
        self._y_prior_proba = np.array(y_prior_proba)
        # 特征 Xi 在所属类别 yk 中出现的总次数
        self._N_yk_xi = np.zeros((np.size(self.classes_), features))
        # 所有特征在 yk 中出现的总次数
        self._N_yk = np.zeros((np.size(self.classes_), 1))
        



if __name__ == "__main__":
    from sklearn.datasets import load_iris
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split
    from sklearn.naive_bayes import MultinomialNB

    test = np.array([[1, 'S'], [1, 'M'], [1, 'M'], [1, 'S'], [1, 'S'], [2, 'S'], [2, 'M'], 
                    [2, 'M'], [2, 'L'], [2, 'L'], [3, 'L'], [3, 'M'], [3, 'M'], [3, 'L'], [3, 'L']])
    iris = load_iris()
    X = iris.data 
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    gnb = MultinomialNB().fit(X_train, y_train)
    print(gnb.score(X_test, y_test))