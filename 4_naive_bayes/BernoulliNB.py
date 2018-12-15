import numpy as np
from sklearn.preprocessing import binarize
from .BaseNB import BaseNaiveBayes
from sklearn.preprocessing import LabelBinarizer

class BernoulliNaiveBayes(BaseNaiveBayes):

    def __init__(self, alpha: float=1.0, binarize: float=.0):
        """
        alpha: 平滑参数, 防止条件概率分母为 0 的情况
        """
        super(BernoulliNaiveBayes, self).__init__()
        self._alpha = alpha
        self._binarize = binarize

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        X: train dataset, shape = (n_samples, n_features)
        y: target, shape = (n_samples, )
        """
        # 0-1 化
        label_binarizer = LabelBinarizer().fit(y)
        Y = label_binarizer.transform(y)
        if Y.shape[1] == 1:
            Y = np.concatenate((1 - Y, Y), axis=1)
        self.classes_ = label_binarizer.classes_
        self._y_prior_log_proba = np.log(Y.sum(axis=0) / Y.shape[0])
        # 将出现的频数变为是否出现过，出现过即为 1
        X = binarize(X, threshold=self._binarize)
        self._N_yk_xi = np.dot(Y.T, X) + self._alpha
        self._N_yk = Y.sum(axis=0) + 2 * self._alpha
        self._features_log_proba = np.log(self._N_yk_xi) - np.log(self._N_yk.reshape(-1, 1))
        return self

    def _joint_log_likelihood(self, X: np.ndarray):
        neg_proba = np.log(1 - np.exp(self._features_log_proba))
        jll = np.dot(X, (self._features_log_proba - neg_proba).T)
        jll += self._y_prior_log_proba + neg_proba.sum(axis=1)
        return jll

    def __repr__(self):
        return "<MultinomialNaiveBayes>" 

if __name__ == "__main__":
    from sklearn.datasets import load_iris
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split
    from sklearn.naive_bayes import BernoulliNB
    from sklearn.feature_extraction.text import CountVectorizer
    import pandas as pd 

    data = open(r'4_naive_bayes\corpus').read()
    labels, texts = [], []
    for line in data.split('\n'):
        content = line.split(maxsplit=1)
        labels.append(content[0][-1])
        texts.append(content[1])

    data = pd.DataFrame(data={'labels': labels, 'texts': texts})
    X_train, X_test, y_train, y_test = train_test_split(data['texts'], data['labels'].apply(int), random_state=0)
    vect = CountVectorizer().fit(X_train)
    X_train_vectorized = vect.transform(X_train)
    X_test_vectorized = vect.transform(X_test)

    mnb = BernoulliNaiveBayes().fit(X_train_vectorized.toarray(), y_train)
    y_pred = mnb.predict(X_test_vectorized.toarray())
    print(accuracy_score(y_test, y_pred))
    print('-----------------------------------')
    mNB = BernoulliNB().fit(X_train_vectorized, y_train)
    print(mNB.score(X_test_vectorized, y_test))
