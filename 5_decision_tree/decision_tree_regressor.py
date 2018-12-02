import numpy as np 

class Node:

    def __init__(self, val: float, pair: tuple):
        self._val = val
        self._best_pair = pair
        self._left = None
        self._right = None

    def __repr__(self):
        return "<Node: %d>" % self._val

class DecisionTreeRegressor:

    def __init__(self):
        self.root = None

    def _divide(self, x: np.ndarray, y: np.ndarray, partition_feature_index: int, partition_point_index: int):
        first_part_index = np.argwhere(x[:, partition_feature_index] <= x[partition_point_index, partition_feature_index])
        second_part_index = np.argwhere(x[:, partition_feature_index] > x[partition_point_index, partition_feature_index])
        return x[first_part_index, :], x[second_part_index, :], y[first_part_index], y[second_part_index]

    def _generate_regression_tree(self, X: np.ndarray, y: np.ndarray):
        rows, features = X.shape
        # 切分特征和切分点索引初始化为 0
        j = s = 0
        best_j, best_s = j, s
        _, _, first_part_y, second_part_y = self._divide(X, y, j, s)
        min_loss = first_part_y.var() + second_part_y.var()
        for j in range(features):
            for s in range(rows):
                _, _, first_part_y, second_part_y = self._divide(X, y, j, s)
                loss = first_part_y.var() + second_part_y.var()
                if loss < min_loss:
                    best_j, best_s = j, s
                    min_loss = loss
        pair = (best_j, best_s)
        root = Node(y.mean(), pair)
        first_part_x, second_part_x, first_part_y, second_part_y = self._divide(X, y, best_j, best_s)
        root.left = self._generate_regression_tree(first_part_x, first_part_y)
        root.right = self._generate_regression_tree(second_part_x, second_part_y)
        return root

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.root = self._generate_regression_tree(X, y)
        return self

    def predict(self, X: np.ndarray):
        pass

