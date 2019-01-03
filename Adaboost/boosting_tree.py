# @Time    : 2019/1/3 9:25
# @Author  : Xu Huipeng
# @Blog    : https://brycexxx.github.io/
import numpy as np
from typing import Tuple
from decision_tree.decision_tree_regressor import Node
from uuid import uuid1
from sklearn.metrics import r2_score


class BoostingSimpleTree:
    """
    弱分类器是一个根节点直接连接
    两个叶节点的简单回归决策树
    """

    def __init__(self, n_estimators: int = 50):
        self.n_estimators = n_estimators

    def _split(self, x: np.ndarray, y: np.ndarray,
               split_feature_index: int, split_point: float) \
            -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        根据特征列和切分点将数据集分割为左右两部分
        """
        left_index = x[:, split_feature_index] <= split_point
        right_index = x[:, split_feature_index] > split_point
        return x[left_index, :], x[right_index, :], y[left_index], y[right_index]

    def _base_tree(self, X: np.ndarray, r: np.ndarray) -> tuple:
        # best_f_p = (feature_index, split_point, left_output, right_output)
        best_f_p = (None, None, None, None)
        min_loss = r.var() * np.size(r)
        for f in range(self.n_features):
            unique_x = np.unique(X[:, f])
            split_point = [(unique_x[i] + unique_x[i + 1]) / 2.0 for i in range(np.size(unique_x) - 1)]
            for p in split_point:
                _, _, left_y, right_y = self._split(X, r, f, p)
                loss = left_y.var() * np.size(left_y) + right_y.var() * np.size(right_y)
                if loss < min_loss:
                    min_loss = loss
                    best_f_p = (f, p, left_y.mean(), right_y.mean())
        return best_f_p

    def fit(self, X: np.ndarray, y: np.ndarray):
        n_samples, self.n_features = X.shape
        self.tree_series = []
        # 残差
        # y = y.reshape(-1, 1)
        r = y.copy()
        for i in range(self.n_estimators):
            best_f_p = self._base_tree(X, r)
            if best_f_p[0] is None:
                break
            self.tree_series.append(best_f_p)
            y_pred = self.predict(X)
            r = y - y_pred
            # squared_error = np.linalg.norm(r) ** 2
            # print('training %d base tree, squared error: %.2f' % (i + 1, squared_error))
            r2 = r2_score(y, y_pred)
            print('training %d base tree, r2-score: %.2f' % (i + 1, r2))
        return self

    def predict(self, X: np.ndarray):
        m = X.shape[0]
        y = np.zeros((m,))
        for i in range(m):
            for f, p, left_output, right_output in self.tree_series:
                output = left_output if X[i, f] < p else right_output
                y[i] += output
        return y


class BoostingDecisionTree:
    """
    以回归决策树作为弱分类器
    """
    def __init__(self, n_estimators: int=50, max_depth: int=3):
        self.n_estimators = n_estimators
        self.max_depth = max_depth

    def _split(self, x: np.ndarray, y: np.ndarray,
               split_feature_index: int, split_point: float) \
            -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        根据特征列和切分点将数据集分割为左右两部分
        """
        left_index = x[:, split_feature_index] <= split_point
        right_index = x[:, split_feature_index] > split_point
        return x[left_index, :], x[right_index, :], y[left_index], y[right_index]

    def _generate_regression_tree(self, X: np.ndarray, y: np.ndarray, depth: int=0):
        """
        递归生成最小二乘回归树
        """
        # 初始化最优分割点
        best_feature = best_point = None
        pair = (best_feature, best_point)
        y_var = y.var()
        min_loss = y_var * np.size(y)
        rows, features = X.shape
        # 如果样本数量少于 2 ，则停止分割，生成叶节点
        if rows < 2:
            return Node(y.mean(), y.var(), rows, pair, str(uuid1()))
        # 如果样本全部属于一个类则停止分割，生成叶节点
        if np.size(np.unique(y)) == 1:
            return Node(y[0], 0, rows, pair, str(uuid1()))
        for f in range(features):
            # 去重
            unique_point = np.unique(X[:, f])
            # 计算相邻元素中值作为分割点
            split_point = [(unique_point[i] + unique_point[i+1]) / 2.0 for i in range(np.size(unique_point) - 1)]
            # 遍历分割点
            for p in split_point:
                _, _, left_y, right_y = self._split(X, y, f, p)
                loss = left_y.var() * np.size(left_y) + right_y.var() * np.size(right_y)
                if loss < min_loss:
                    best_feature, best_point = f, p
                    min_loss = loss
        pair = (best_feature, best_point)
        root = Node(y.mean(), y_var, rows, pair, str(uuid1()))
        # 如果遍历完没找到最优分割特征，则停止分割，生成叶节点, 配合预剪枝
        if best_feature is None:
            return root
        left_x, right_x, left_y, right_y = self._split(X, y, best_feature, best_point)
        depth += 1
        if depth <= self.max_depth:
            root._left = self._generate_regression_tree(left_x, left_y, depth)
            root._right = self._generate_regression_tree(right_x, right_y, depth)
        return root

    def fit(self,X: np.ndarray, y: np.ndarray):
        self.tree_series = []
        r = y.copy()
        for i in range(self.n_estimators):
            depth = 0
            root = self._generate_regression_tree(X, r, depth)
            self.tree_series.append(root)
            y_pred = self.predict(X)
            r = y - y_pred
            r2 = r2_score(y, y_pred)
            print('training %d decision tree, r2-score: %.2f' % (i + 1, r2))
        return self

    def _predict_on_one_tree(self, X: np.ndarray, root: Node):
        """
        X: data to be predicted, shape = (n_samples, n_features)
        """
        ret = []
        for x in X:
            split_feature, split_point = root._best_pair
            node = root
            while node._left != None:
                if x[split_feature] <= split_point:
                    node = node._left
                else:
                    node = node._right
                split_feature, split_point = node._best_pair
            ret.append(node._val)
        return np.array(ret)

    def predict(self, X: np.ndarray):
        m = X.shape[0]
        y_pred = np.zeros((m, ))
        for root in self.tree_series:
            output = self._predict_on_one_tree(X, root)
            y_pred += output
        return y_pred


if __name__ == "__main__":
    # 统计学习方法算例
    # x = np.array(list(range(1, 11))).reshape(-1, 1)
    # y = np.array([5.56, 5.70, 5.91, 6.40, 6.80, 7.05, 8.90, 8.70, 9.00, 9.05])
    #
    # bt = BoostingTree(n_estimators=6)
    # bt.fit(x, y)

    from sklearn.datasets import load_boston
    from sklearn.metrics import r2_score
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import AdaBoostRegressor
    from sklearn.tree import DecisionTreeRegressor

    boston = load_boston()
    X, y = boston.data, boston.target
    train_X, test_X, train_y, test_y = train_test_split(X, y, random_state=0)
    bt = BoostingSimpleTree(n_estimators=18).fit(train_X, train_y)
    y_pred = bt.predict(test_X)
    print("boosting simple tree,  r2-score on test set: %.2f" % r2_score(test_y, y_pred))
    abr = AdaBoostRegressor(
        DecisionTreeRegressor(max_depth=3, min_samples_split=20, min_samples_leaf=5),
        loss='linear', n_estimators=800, learning_rate=0.5, random_state=0
    ).fit(train_X, train_y)
    print("AdaBoostRegressor of sklearn, r2-score on test set: %.2f" % abr.score(test_X, test_y))
    print('-' * 10)
    bdt = BoostingDecisionTree(n_estimators=20, max_depth=5).fit(train_X, train_y)
    print("boosting decision tree,  r2-score on test set: %.2f"% r2_score(test_y, bdt.predict(test_X)))