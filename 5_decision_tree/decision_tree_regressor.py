import numpy as np 
from typing import Tuple

class Node:

    def __init__(self, val: float, mse: float, samples: int, pair: tuple):
        self._val = val
        self._mse = mse
        self._samples = samples
        self._best_pair = pair
        self._left = None
        self._right = None

    def __repr__(self):
        return "<Node: %d>" % self._val

class DTRegressor:

    def __init__(self):
        self.root = None

    def _split(self, x: np.ndarray, y: np.ndarray, split_feature_index: int, split_point: float):
        left_index = x[:, split_feature_index] <= split_point
        right_index = x[:, split_feature_index] > split_point
        return x[left_index, :], x[right_index, :], y[left_index], y[right_index]

    def chose_best_feature(self, x: np.ndarray, y: np.ndarray, op: Tuple[int]=(1, 2)):
        if np.size(np.unique(y)) == 1:
            return None, y.mean()
        y_var = y.var() * np.size(y)
        best_feature = -1
        best_point = 0
        low_error = np.inf
        m, n = x.shape
        for i in range(n):
            for i in np.unique(y):
                _, _, left_y, right_y = self._split(x, y, i, i)
                if np.size(left_y) < op[1] or np.size(right_y) < op[1]: continue
                temp_error = left_y.var() * np.size(left_y) + right_y.var() * np.size(right_y)
                if temp_error < low_error:
                    low_error = temp_error
                    best_feature = i
                    best_point = i
        if y_var - low_error < op[0]:
            return None, y.mean()
        _, _, left_y, right_y = self._split(x, y, best_feature, best_point)
        return best_feature, best_point       

    def _generate_regression_tree(self, X: np.ndarray, y: np.ndarray):
        best_feature = best_point = None
        pair = (best_feature, best_point)
        y_var = y.var()
        min_loss = y_var * np.size(y)
        rows, features = X.shape
        if rows < 2:
            return Node(y.mean(), y.var(), rows, pair)
        if np.size(np.unique(y)) == 1:
            return Node(y[0], 0, rows, pair)
        for f in range(features):
            unique_point = np.unique(X[:, f])
            split_point = [(unique_point[i] + unique_point[i+1]) / 2 for i in range(np.size(unique_point) - 1)]
            for p in split_point:
                _, _, left_y, right_y = self._split(X, y, f, p)
                left_var = left_y.var() * np.size(left_y) if np.size(left_y) else 0
                right_var = right_y.var() * np.size(right_y) if np.size(right_y) else 0
                loss = left_var + right_var
                if loss < min_loss:
                    best_feature, best_point = f, p
                    min_loss = loss
        pair = (best_feature, best_point)
        root = Node(y.mean(), y_var, rows, pair)
        if best_feature is None:
            return root
        left_x, right_x, left_y, right_y = self._split(X, y, best_feature, best_point)
        root._left = self._generate_regression_tree(left_x, left_y)
        root._right = self._generate_regression_tree(right_x, right_y)
        return root

    # def _gt(self, x, y):
    #     best_f, best_pnt = self.chose_best_feature(x, y)
    #     pair = (best_f, best_pnt)
    #     if best_f == None: return Node(best_pnt, (None, None))
    #     data_l_x, data_r_x, data_l_y, data_r_y = self._split(x, y, best_f, best_pnt)
    #     root = Node(y.mean(), pair)
    #     root._left = self._gt(data_l_x, data_l_y)
    #     root._right = self._gt(data_r_x, data_r_y)
    #     return root

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.root = self._generate_regression_tree(X, y)
        return self

    def predict(self, X: np.ndarray):
        split_feature, split_point = self.root._best_pair
        node = self.root
        ret = []
        for x in X:
            while split_feature:
                if x[split_feature] <= split_point:
                    node = node._left
                else:
                    node = node._right
                split_feature, split_point = node._best_pair
            ret.append(node._val)
        return np.array(ret)

if __name__ == "__main__":
    from sklearn.datasets import load_boston
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.tree import DecisionTreeRegressor, export_graphviz
    import graphviz

    boston = load_boston()
    X = boston.data
    y = boston.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    dtr = DTRegressor().fit(X_train, y_train)
    root = dtr.root
    print((root._best_pair[1], root._mse, root._samples, root._val))
    children = [root._left, root._right]
    while children:
        child = children.pop(0)
        print((child._best_pair[1], child._mse, child._samples, child._val))
        if child._left:
            children.append(child._left)
        if child._right:
            children.append(child._right)
    
    y_pred = dtr.predict(X_test)
    print(mean_squared_error(y_test, y_pred))