import numpy as np
from graphviz import Digraph
from copy import deepcopy
from uuid import uuid1
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

class Node:
    def __init__(self, val: float, mse: float, samples: int, pair: tuple, id_: str):
        # 节点所包含的样本的均值
        self._val = val
        # 节点所包含样本的均方差
        self._mse = mse
        # 节点所包含样本的数量
        self._samples = samples
        # 节点的最优分割点
        self._best_pair = pair
        self._id = id_
        self._left = None
        self._right = None

    def __repr__(self):
        return "<Node: %d>" % self._val


class DTRegressor:

    def __init__(self, is_pre_pruning: bool=True):
        self.root = None
        self._is_pre_pruning = is_pre_pruning

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
        # 是否进行预剪枝
        min_loss = y_var * np.size(y) if self._is_pre_pruning else np.inf
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
        root = Node(y.mean(), y_var, rows, pair, str(uuid1()))
        # 如果遍历完没找到最优分割特征，则停止分割，生成叶节点, 配合预剪枝
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
        # 后剪枝
        if not self._is_pre_pruning:
            X_train, X_validate, y_train, y_validate = train_test_split(X, y, test_size=0.1, random_state=0)
            T_0 = self._generate_regression_tree(X_train, y_train)
            sub_trees_series = self._poster_pruning(T_0)
            st_scores = []
            alphas = []
            for sub_tree in sub_trees_series:
                self.root = sub_tree[1]
                self.plot_tree()
                alphas.append(sub_tree[0])
                st_score = self.score(X_validate, y_validate)
                st_scores.append(st_score)
            score_and_alpha = list(zip(st_scores, alphas, range(len(st_scores))))
            score_and_alpha.sort(reverse=True)
            optimal_index = score_and_alpha[0][-1]
            self.root = sub_trees_series[optimal_index][1]
        # 预剪枝
        else:
            self.root = self._generate_regression_tree(X, y)
        return self

    def _get_pruned_tree(self, root: Node):
        alpha = np.inf
        # 只有根节点
        if not any((root._left, root._right)):
            return (alpha, root)
        children = [root]
        min_gt_node = root
        while children:
            child = children.pop(0)
            # 跳过叶节点
            if child._left._left:
                children.append(child._left)
            if child._right._left:
                children.append(child._right)
            # 剪枝后的预测误差
            c_t = child._mse * child._samples
            # 内部节点的叶结点个数
            leaf_nums = 0
            c_T_t = 0
            child_sub_node = [child._left, child._right]
            while child_sub_node:
                sub_node = child_sub_node.pop(0)
                if not any((sub_node._left, sub_node._right)):
                    # 剪枝前预测误差
                    c_T_t += sub_node._mse * sub_node._samples
                    leaf_nums += 1
                    continue
                child_sub_node.append(sub_node._left)
                child_sub_node.append(sub_node._right)
            # 剪枝阈值
            g_t = (c_t - c_T_t) / (leaf_nums - 1)
            if g_t < alpha:
                alpha = g_t
                min_gt_node = child
        # 将内部节点转换为叶节点
        if min_gt_node != root:
            min_gt_node._left = None
            min_gt_node._right = None
            min_gt_node._best_pair = (None, None)
        return (alpha, root)

    def _poster_pruning(self, root: Node):
        """
        对已经生成的决策树进行剪枝得到最优子树序列
        """
        sub_trees_series = [(np.inf, root)]
        pruned_tree = root
        # 构造最优子树序列
        while True:
            alpha, pruned_tree = self._get_pruned_tree(deepcopy(pruned_tree))
            sub_trees_series.append((alpha, pruned_tree))
            if not any((pruned_tree._left._best_pair[0], pruned_tree._right._best_pair[0])):
                break
        return sub_trees_series

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

    def score(self, X: np.ndarray, y: np.ndarray):
        y_pred = self.predict(X)
        u = np.sum((y - y_pred) ** 2)
        v = np.sum((y - y.mean()) ** 2)
        return 1 - u / v

    def get_params(self):
        return {'is_pre_pruning': self._is_pre_pruning}

    def _get_name_label(self, node: Node):
        label = """
            x[{feature}] <= {split_point}
            mse = {mse}
            samples = {samples}
            values = {values}
            """.format(
            feature=node._best_pair[0],
            split_point=node._best_pair[1],
            mse=node._mse,
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
        tree.render('decision_tree.gv', view=True)


if __name__ == "__main__":
    from sklearn.datasets import load_boston
    from sklearn.model_selection import train_test_split, cross_validate
    from sklearn.metrics import mean_squared_error, r2_score

    boston = load_boston()
    X = boston.data
    y = boston.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    # dtr = DTRegressor()
    # ret = cross_validate(dtr, X_train, y_train, cv=3)
    # print(ret)
    # dtr = DTRegressor(is_pre_pruning=False).fit(X_train, y_train)
    dtr = DTRegressor().fit(X_train, y_train)
    dtr.plot_tree()
    print(dtr.score(X_test, y_test))
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
    
    # y_pred = dtr.predict(X_test)
    # print(y_pred)
    # print(mean_squared_error(y_test, y_pred))
    # test = X_train[(X_train[:, 12] <= 8.13) & (X_train[:, 5] > 7.435) & (X_train[:, 10] <= 18.3) & (X_train[:, 0] > 0.577)]
    # print(dtr.predict(test))