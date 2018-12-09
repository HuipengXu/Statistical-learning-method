# @Time    : 2018/12/9 18:32
# @Author  : Xu Huipeng
# @Blog    : https://brycexxx.github.io/

import numpy as np
from uuid import uuid1
from sklearn.preprocessing import label_binarize

class Node:

    def __init__(self, val: list, gini: np.float64, samples: int, pair: tuple, id_: str):
        self._val = val
        self._gini = gini
        self._samples = samples
        self._best_pair = pair
        self._id = id_
        self._left = None
        self._right = None

    def __repr__(self):
        return "<Node: %s>" % str(self._val)

class DTClassifier:

    def __init__(self, samples_threshold: int=2, gini_threshold: float=.0, is_pre_pruning: bool=True):
        self.root = None
        self._samples_threshold = samples_threshold
        self._gini_threshold = gini_threshold
        self._is_pre_pruning = is_pre_pruning

    def _calculate_gini(self, y: np.ndarray):
        y = label_binarize(y, classes=np.unique(y))
        if y.shape[1] == 1:
            y = np.concatenate((1 - y, y), axis=1)
        samples = y.shape[0]
        proba_k = y.sum(axis=0) / samples
        gini = 1 - np.sum(proba_k ** 2)
        return gini, y.sum(axis=0).tolist()

    def _split(self, x: np.ndarray, y: np.ndarray, split_feature: int, split_point: float):
        left_index = x[:, split_feature] <= split_point
        right_index = x[:, split_feature] > split_point
        return x[left_index], x[right_index], y[left_index], y[right_index]

    def _generate_classification_tree(self, X: np.ndarray, y: np.ndarray):
        rows, features = X.shape
        best_feature = best_point = None
        pair = (best_feature, best_point)
        gini, val = self._calculate_gini(y)
        min_gini = gini if self._is_pre_pruning else 1.0
        if rows < self._samples_threshold or gini <= self._gini_threshold :
            return Node(val, gini, rows, pair, str(uuid1()))
        for f in range(features):
            unique_point = np.unique(X[:, f])
            split_point = [(unique_point[i] + unique_point[i+1]) / 2 for i in range(np.size(unique_point) - 1)]
            split_point.insert(0, unique_point[0])
            split_point.append(unique_point[-1])
            for p in split_point:
                _, _, left_y, right_y = self._split(X, y, f, p)
                gini_left, _ = self._calculate_gini(left_y)
                gini_right, _ = self._calculate_gini(right_y)
                gini_f = (np.size(left_y) * gini_left + np.size(right_y) * gini_right) / rows
                if gini_f < min_gini:
                    min_gini = gini_f
                    best_feature, best_point = f, p
        pair = (best_feature, best_point)


    def fit(self, X: np.ndarray, y: np.ndarray):
        pass

if __name__ == "__main__":
    from sklearn.tree import DecisionTreeClassifier, export_graphviz
    from sklearn.model_selection import train_test_split
    from sklearn.datasets import load_iris
    import graphviz

    iris = load_iris()
    X = iris.data
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    clf = DecisionTreeClassifier().fit(X_train, y_train)
    data = export_graphviz(clf)
    graph = graphviz.Source(data)
    graph.render('clf')
