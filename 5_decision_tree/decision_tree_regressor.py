import numpy as np 
from typing import Tuple

class Node:

    def __init__(self, val: float, mse: float, samples: int, pair: tuple):
        # 节点所包含的样本的均值
        self._val = val
        # 节点所包含样本的均方差
        self._mse = mse
        # 
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
        """
        根据特征列和切分点将数据集分割为左右两部分
        """
        left_index = x[:, split_feature_index] <= split_point
        right_index = x[:, split_feature_index] > split_point
        return x[left_index, :], x[right_index, :], y[left_index], y[right_index]    

    def _generate_regression_tree(self, X: np.ndarray, y: np.ndarray):
        """
        递归生成最小二乘回归树
        """
        # 初始化最有分割点
        best_feature = best_point = None
        pair = (best_feature, best_point)
        y_var = y.var()
        min_loss = y_var * np.size(y)
        rows, features = X.shape
        # 如果样本数量少于 2 ，则停止分割，生成叶节点
        if rows < 2:
            return Node(y.mean(), y.var(), rows, pair)
        # 如果样本全部属于一个类则停止分割，生成叶节点
        if np.size(np.unique(y)) == 1:
            return Node(y[0], 0, rows, pair)
        for f in range(features):
            # 去重
            unique_point = np.unique(X[:, f])
            # 计算相邻元素中值作为分割点
            split_point = [(unique_point[i] + unique_point[i+1]) / 2 for i in range(np.size(unique_point) - 1)]
            # 添加第一个和最后一个作为分割点
            split_point.insert(0, unique_point[0])
            split_point.append(unique_point[-1])
            # 遍历分割点
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
        # 如果遍历完没找到最优分割特征，则停止分割，生成叶节点
        if best_feature is None:
            return root
        left_x, right_x, left_y, right_y = self._split(X, y, best_feature, best_point)
        root._left = self._generate_regression_tree(left_x, left_y)
        root._right = self._generate_regression_tree(right_x, right_y)
        return root

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        X, y: train datasets
        X.shape = (n_samples, n_features)
        y.shape = (n_samples, )

        获得决策树根节点
        """
        self.root = self._generate_regression_tree(X, y)
        return self

    def _poster_pruning(self):
        """
        对已经生成的决策树进行剪枝
        """
        

    def predict(self, X: np.ndarray):
        """
        X: data to be predicted, shape = (n_samples, n_features)
        """
        ret = []
        for x in X:
            split_feature, split_point = self.root._best_pair
            node = self.root
            while split_feature != None:
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
    # 广度遍历
    # root = dtr.root
    # print((root._best_pair, root._mse, root._samples, root._val))
    # children = [root._left, root._right]
    # leaf = []
    # while children:
    #     child = children.pop(0)
    #     if not any((child._left, child._right)):
    #         leaf.append(child)
    #     # print((child._best_pair, child._mse, child._samples, child._val))
    #     if child._left:
    #         children.append(child._left)
    #     if child._right:
    #         children.append(child._right)

    # for l in leaf:
    #     print((l._best_pair, l._mse, l._samples, l._val))
    
    y_pred = dtr.predict(X_test)
    # print(y_pred)
    print(mean_squared_error(y_test, y_pred))
    # test = X_train[(X_train[:, 12] <= 8.13) & (X_train[:, 5] > 7.435) & (X_train[:, 10] <= 18.3) & (X_train[:, 0] > 0.577)]
    # print(dtr.predict(test))