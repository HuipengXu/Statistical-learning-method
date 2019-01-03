# @Time    : 2018/12/9 18:32
# @Author  : Xu Huipeng
# @Blog    : https://brycexxx.github.io/

import numpy as np
from uuid import uuid1
from graphviz import Digraph
from sklearn.preprocessing import LabelBinarizer

class Node:

    def __init__(self, val: list, uncertainty: np.float64, samples: int, pair: tuple, id_: str):
        self._val = val
        self._uncertainty = uncertainty
        self._samples = samples
        self._best_pair = pair
        self._id = id_
        self._left = None
        self._right = None

    def __repr__(self):
        return "<Node: %s>" % str(self._val)

class DTClassifier:

    def __init__(self, criterion: str='gini', samples_threshold: int=2, uncertainty_threshold: np.float64=.0, is_pre_pruning: bool=True):
        self.root = None
        self.classes = None
        self._criterion = criterion
        self._samples_threshold = samples_threshold
        self._uncertainty_threshold = uncertainty_threshold
        self._is_pre_pruning = is_pre_pruning

    def _calculate_gini(self, y: np.ndarray):
        if np.size(y) == 0:
            return .0, [0] * np.size(self.binarizer.classes_)
        y = self.binarizer.transform(y)
        if y.shape[1] == 1:
            y = np.concatenate((1 - y, y), axis=1)
        samples = y.shape[0]
        proba_k = y.sum(axis=0) / samples
        gini = 1 - np.sum(proba_k ** 2)
        return gini, y.sum(axis=0).tolist()

    def _calculate_entropy(self, y: np.ndarray):
        if np.size(y) == 0:
            return .0, [0] * np.size(self.binarizer.classes_)
        y = self.binarizer.transform(y)
        if y.shape[1] == 1:
            y = np.concatenate((1 - y, y), axis=1)
        samples = y.shape[0]
        proba_k = y.sum(axis=0) / samples
        log2_proba = np.log2(proba_k+1e-9)
        entropy = -np.sum(np.multiply(proba_k, log2_proba))
        return np.where(entropy<=0, .0, entropy), y.sum(axis=0).tolist()

    def _split(self, x: np.ndarray, y: np.ndarray, split_feature: int, split_point: float):
        left_index = x[:, split_feature] <= split_point
        right_index = x[:, split_feature] > split_point
        return x[left_index], x[right_index], y[left_index], y[right_index]

    def _generate_classification_tree(self, X: np.ndarray, y: np.ndarray, calculate_uncertainty):
        rows, features = X.shape
        best_feature = best_point = None
        pair = (best_feature, best_point)
        uncertainty, val = calculate_uncertainty(y)
        min_uncertainty = uncertainty if self._is_pre_pruning else np.inf
        if rows < self._samples_threshold or uncertainty <= self._uncertainty_threshold:
            return Node(val, min_uncertainty, rows, pair, str(uuid1()))
        for f in range(features):
            unique_point = np.unique(X[:, f])
            split_point = [(unique_point[i] + unique_point[i+1]) / 2 for i in range(np.size(unique_point) - 1)]
            split_point.insert(0, unique_point[0])
            split_point.append(unique_point[-1])
            for p in split_point:
                _, _, left_y, right_y = self._split(X, y, f, p)
                uncertainty_left, _ = calculate_uncertainty(left_y)
                uncertainty_right, _ = calculate_uncertainty(right_y)
                uncertainty_f = (np.size(left_y) * uncertainty_left + np.size(right_y) * uncertainty_right) / rows
                if uncertainty_f < min_uncertainty:
                    min_uncertainty = uncertainty_f
                    best_feature, best_point = f, p
        pair = (best_feature, best_point)
        root = Node(val, uncertainty, rows, pair, str(uuid1()))
        if best_feature == None:
            return root
        left_x, right_x, left_y, right_y = self._split(X, y, best_feature, best_point)
        root._left = self._generate_classification_tree(left_x, left_y, calculate_uncertainty)
        root._right = self._generate_classification_tree(right_x, right_y, calculate_uncertainty)
        return root

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.binarizer = LabelBinarizer().fit(y)
        if self._criterion == 'gini':
            calculate_uncertainty = self._calculate_gini
        elif self._criterion == 'entropy':
            calculate_uncertainty = self._calculate_gini
        else:
            raise KeyError('%s' % self._criterion)
        self.root = self._generate_classification_tree(X, y, calculate_uncertainty)
        return self

    def predict(self, X: np.ndarray):
        ret = []
        for x in X:
            node = self.root
            best_feature, best_point = node._best_pair
            while best_feature != None:
                if x[best_feature] <= best_point:
                    node = node._left
                else:
                    node = node._right
                best_feature, best_point = node._best_pair
            ret.append(node._val)
        ret = np.array(ret)
        return ret.argmax(axis=1)

    def _get_name_label(self, node: Node):
        label = """
            x[{feature}] <= {split_point}
            {criterion} = {uncertainty_val}
            samples = {samples}
            values = {values}
            """.format(
            feature=node._best_pair[0],
            split_point=node._best_pair[1],
            criterion=self._criterion,
            uncertainty_val=node._uncertainty,
            samples=node._samples,
            values=node._val
        )
        name = str(node._id)
        return name, label

    def plot_tree(self):
        tree = Digraph(name='decision_tree', node_attr={'shape': 'square'})
        children = [self.root]
        parent_name, parent_label = self._get_name_label(self.root)
        tree.node(parent_name, label=parent_label)
        while children:
            node = children.pop(0)
            parent_name, _ = self._get_name_label(node)
            left_child_name, left_child_label = self._get_name_label(node._left)
            right_child_name, right_child_label = self._get_name_label(node._right)
            tree.node(left_child_name, label=left_child_label)
            tree.node(right_child_name, label=right_child_label)
            tree.edge(parent_name, left_child_name, 'true')
            tree.edge(parent_name, right_child_name, 'false')
            if node._left._left:
                children.append(node._left)
            if node._right._left:
                children.append(node._right)
        tree.render('decision_tree_classifier.gv', view=True)

if __name__ == "__main__":
    from sklearn.tree import DecisionTreeClassifier, export_graphviz
    from sklearn.model_selection import train_test_split
    from sklearn.datasets import load_iris, load_wine
    from sklearn.metrics import accuracy_score
    import graphviz

    iris = load_iris()
    X = iris.data
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    clf = DTClassifier(criterion='gini').fit(X_train, y_train)
    clf.plot_tree()
    y_pred = clf.predict(X_test)
    print(accuracy_score(y_test, y_pred))
    # clf = DecisionTreeClassifier(criterion='entropy').fit(X_train, y_train)
    # data =export_graphviz(clf)
    # graph = graphviz.Source(data)
    # graph.render('clf_entropy')

