# @Time    : 2019/3/8 10:16
# @Author  : Xu Huipeng
# @Blog    : https://brycexxx.github.io/
from typing import Tuple
from sklearn.preprocessing import LabelBinarizer
from decision_tree.decision_tree_regressor import Node
from uuid import uuid1
import matplotlib.pyplot as plt
import numpy as np


class GBDT:

    def __init__(self, n_estimators: int = 100, learning_rate: float = 0.1):
        self.n_estimators = n_estimators
        self.lr = learning_rate
        self.base = []

    def _get_neg_grad(self, y: np.ndarray, f: np.ndarray):
        """
        使用类似于逻辑回归的对数似然损失函数
        """
        return y / (1 + np.exp(y * f))

    def _split(self, x: np.ndarray, y: np.ndarray, split_fea_idx: int, split_point: float):
        """
        根据特征列和切分点将数据集分割为左右两部分
        """
        left_index = x[:, split_fea_idx] <= split_point
        right_index = x[:, split_fea_idx] > split_point
        return x[left_index, :], x[right_index, :], y[left_index], y[right_index]

    def _get_c_star(self, X: np.ndarray, y: np.ndarray):
        f = self._predict(X)
        r = self._get_neg_grad(y, f)
        c_star = r.sum() / ((np.abs(r) * (1 - np.abs(r))).sum() + 1e-9)
        return c_star

    def _build_cart(self, X: np.ndarray, y: np.ndarray):
        # 初始化最有分割点
        best_feature = best_point = None
        pair = (best_feature, best_point)
        y_var = y.var()
        min_loss = y_var * np.size(y)
        rows, features = X.shape
        # 如果样本数量少于 2 ，则停止分割，生成叶节点；或者样本全部属于一个类则停止分割，生成叶节点
        if rows < 2 or np.size(np.unique(y)) == 1:
            c_star = self._get_c_star(X, y)
            return Node(c_star, y.var(), rows, pair, str(uuid1()))
        for f in range(features):
            # 去重
            unique_point = np.unique(X[:, f])
            # 计算相邻元素中值作为分割点
            split_point = [(unique_point[i] + unique_point[i + 1]) / 2 for i in range(np.size(unique_point) - 1)]
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
        # 如果遍历完没找到最优分割特征，则停止分割，生成叶节点, 配合预剪枝
        if best_feature is None:
            c_star = self._get_c_star(X, y)
            return Node(c_star, y.var(), rows, pair, str(uuid1()))
        root = Node(y.mean(), y_var, rows, pair, str(uuid1()))
        left_x, right_x, left_y, right_y = self._split(X, y, best_feature, best_point)
        root._left = self._build_cart(left_x, left_y)
        root._right = self._build_cart(right_x, right_y)
        return root

    def fit(self, X: np.ndarray, y: np.ndarray):
        f = y.mean()
        self.base.append(f)
        for i in range(self.n_estimators):
            r = self._get_neg_grad(y, f)
            root = self._build_cart(X, r)
            print('{:d} estimators have been trained'.format(i + 1))
            self.base.append(root)
            f = self._predict(X)
        return self

    def sigmoid(self, x: np.ndarray):
        return 1 / (1 + np.exp(-x))

    def _predict(self, X: np.ndarray):
        ret = []
        for x in X:
            f = self.base[0]
            for node in self.base[1:]:
                split_fea_idx, split_point = node._best_pair
                while split_fea_idx != None:
                    if x[split_fea_idx] <= split_point:
                        node = node._left
                    else:
                        node = node._right
                    split_fea_idx, split_point = node._best_pair
                f += self.lr * node._val
            ret.append(f)
        return self.sigmoid(np.array(ret))

    def predict(self, X: np.ndarray):
        ret = self._predict(X)
        ret = np.where(ret >= 0.5, 1, -1)
        return ret


if __name__ == "__main__":
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import *
    from sklearn.preprocessing import LabelBinarizer
    from sklearn.ensemble import GradientBoostingClassifier

    bd = load_breast_cancer()
    X = bd.data
    y = bd.target
    y = LabelBinarizer(neg_label=-1, pos_label=1).fit_transform(y).squeeze()
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, random_state=42)
    gbdt = GBDT(n_estimators=150, learning_rate=0.06)
    gbdt.fit(X_train, y_train)
    y_preds = gbdt.predict(X_valid)
    y_train_preds = gbdt.predict(X_train)
    print('the accuracy on validset is {:.4f}'.format(accuracy_score(y_valid, y_preds)))
    print('the accuracy on trainset is {:.4f}'.format(accuracy_score(y_train, y_train_preds)))
    print('----------------------------------------')
    gbdt1 = GradientBoostingClassifier(random_state=2019, verbose=10)
    gbdt1.fit(X_train, y_train)
    print(gbdt1.score(X_train, y_train))
    print(gbdt1.score(X_valid, y_valid))
