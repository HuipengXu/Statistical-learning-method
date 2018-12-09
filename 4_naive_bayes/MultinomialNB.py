import numpy as np 
from sklearn.preprocessing import  LabelBinarizer
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
        # 0-1 化
        label_binarizer = LabelBinarizer().fit(y) 
        # Y.shape = (n_samples, classes_)
        Y = label_binarizer.transform(y)
        # 处理二分类的时候返回一列的情况
        if Y.shape[1] == 1:
            Y = np.concatenate((1 - Y, Y), axis=1)
        self.classes_ = label_binarizer.classes_
        # self._y_prior_log_proba.shape = (classes_, )
        self._y_prior_log_proba = np.log(Y.sum(axis=0) / Y.shape[0])
        # self._N_yk_xi.shape = (classes_, n_features)
        self._N_yk_xi = np.dot(Y.T, X) + self._alpha
        self._N_yk = self._N_yk_xi.sum(axis=1)
        self._features_log_proba = np.log(self._N_yk_xi) - np.log(self._N_yk.reshape(-1, 1)) 
        return self

    def _joint_log_likelihood(self, X: np.ndarray):
        jll = np.dot(X, self._features_log_proba.T) + self._y_prior_log_proba
        return jll

    def __repr__(self):
        return "<MultinomialNaiveBayes>" 

if __name__ == "__main__":
    from sklearn.datasets import load_iris
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.feature_extraction.text import CountVectorizer
    import pandas as pd 

    X = np.array([[1, 'S'], [1, 'M'], [1, 'M'], [1, 'S'], [1, 'S'], [2, 'S'], [2, 'M'], 
                    [2, 'M'], [2, 'L'], [2, 'L'], [3, 'L'], [3, 'M'], [3, 'M'], [3, 'L'], [3, 'L']])
    y = np.array([-1, -1, 1, 1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, -1])

    data = open(r'4_naive_bayes\corpus').read()
    labels, texts = [], []
    for line in data.split('\n'):
        content = line.split(maxsplit=1)
        labels.append(content[0][-1])
        texts.append(content[1])

    data = pd.DataFrame(data={'labels': labels, 'texts': texts})
    X_train, X_test, y_train, y_test = train_test_split(data['texts'], data['labels'].apply(int), random_state=0)
    vect = CountVectorizer(ngram_range=(1, 3), min_df=3, analyzer='char_wb').fit(X_train)
    X_train_vectorized = vect.transform(X_train)
    X_test_vectorized = vect.transform(X_test)

    mnb = MultinomialNaiveBayes().fit(X_train_vectorized.toarray(), y_train)
    y_pred = mnb.predict(X_test_vectorized.toarray())
    print(accuracy_score(y_test, y_pred))
    print('-----------------------------------')
    mNB = MultinomialNB().fit(X_train_vectorized, y_train)
    print(mNB.score(X_test_vectorized, y_test))