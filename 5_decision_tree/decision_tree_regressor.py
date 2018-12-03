import numpy as np 

class Node:

    def __init__(self, val: float, pair: tuple):
        self._val = val
        self._best_pair = pair
        self._left = None
        self._right = None

    def __repr__(self):
        return "<Node: %d>" % self._val

class DTRegressor:

    def __init__(self):
        self.root = None

    def _divide(self, x: np.ndarray, y: np.ndarray, partition_feature_index: int, partition_point_index: int):
        first_part_index = np.argwhere(x[:, partition_feature_index] <= x[partition_point_index, partition_feature_index]).squeeze()
        second_part_index = np.argwhere(x[:, partition_feature_index] > x[partition_point_index, partition_feature_index]).squeeze()
        return x[first_part_index, :].reshape(-1, x.shape[1]), x[second_part_index, :].reshape(-1, x.shape[1]), y[first_part_index], y[second_part_index]

    def _generate_regression_tree(self, X: np.ndarray, y: np.ndarray):
        print(X.shape)
        rows, features = X.shape
        # 切分特征和切分点索引初始化为 0
        j = s = 0
        best_j, best_s = j, s
        _, _, first_part_y, second_part_y = self._divide(X, y, j, s)
        min_loss = first_part_y.var() + second_part_y.var()
        if min_loss < 1e-3:
            return None
        for j in range(features):
            for s in range(rows):
                _, _, first_part_y, second_part_y = self._divide(X, y, j, s)
                print(first_part_y.shape)
                print(second_part_y.shape)
                print('----------------')
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

if __name__ == "__main__":
    from sklearn.datasets import load_boston
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    from sklearn.tree import DecisionTreeRegressor, export_graphviz
    import graphviz

    boston = load_boston()
    X = boston.data
    y = boston.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    dtr = DTRegressor().fit(X_train, y_train)
