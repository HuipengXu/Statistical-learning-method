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
        first_part_index = x[:, partition_feature_index] <= x[partition_point_index, partition_feature_index]
        second_part_index = x[:, partition_feature_index] > x[partition_point_index, partition_feature_index]
        return x[first_part_index, :], x[second_part_index, :], y[first_part_index], y[second_part_index]

    def _generate_regression_tree(self, X: np.ndarray, y: np.ndarray):
        best_j = best_s = None
        pair = (best_j, best_s)
        min_loss = y.var() * np.size(y)
        rows, features = X.shape
        if rows < 2:
            return Node(y.mean(), pair)
        if np.size(np.unique(y)) == 1:
            return Node(y[0], pair)
        if min_loss < 1e-5:
            return Node(y.mean(), pair)
        for j in range(features):
            for s in range(rows):
                _, _, first_part_y, second_part_y = self._divide(X, y, j, s)
                first_var = first_part_y.var() * np.size(first_part_y) if np.size(first_part_y) else 0
                second_var = second_part_y.var() * np.size(second_part_y) if np.size(second_part_y) else 0
                loss = first_var + second_var
                if loss < min_loss:
                    # print((j, X[s, j]))
                    best_j, best_s = j, s
                    min_loss = loss
        # print('-----------------------------------')
        pair = (best_j, best_s)
        root = Node(y.mean(), pair)
        if not best_j:
            return root
        first_part_x, second_part_x, first_part_y, second_part_y = self._divide(X, y, best_j, best_s)
        root._left = self._generate_regression_tree(first_part_x, first_part_y)
        root._right = self._generate_regression_tree(second_part_x, second_part_y)
        return root

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.root = self._generate_regression_tree(X, y)
        return self

    def predict(self, X: np.ndarray):
        

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
    root = dtr.root
    print(root._val)
    children = [root._left, root._right]
    while children:
        child = children.pop(0)
        print(child)
        if child._left:
            children.append(child._left)
        if child._right:
            children.append(child._right)
    
