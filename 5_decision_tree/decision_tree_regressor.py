import numpy as np 
from typing import Tuple
from sklearn.model_selection import train_test_split
from sklarn.metrics import mean_squared_error

class Node:

    def __init__(self, val: float, mse: float, samples: int, pair: tuple):
        # 节点所包含的样本的均值
        self._val = val
        # 节点所包含样本的均方差
        self._mse = mse
        # 节点所包含样本的数量
        self._samples = samples
        # 节点的最优分割点
        self._best_pair = pair
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
        if self._is_pre_pruning
            min_loss = y_var * np.size(y)  
        else:
            min_loss = np.inf
            # 为后剪枝留出独立验证集
            X, self._X_validation, y, self._y_validation = train_test_split(X, y, test_size=0.1, random_state=0)
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
        self.root = self._generate_regression_tree(X, y)
        return self

    def _get_pruned_tree(self, X: np.ndarray, y: np.ndarray, root: Node):
        """"""
        alpha = np.inf
        # 只有根节点
        if not any((root._left, root._right)):
            return (alpha, root)
        children = [root._left, root._right]
        min_gt_tree = root
        while children:
            child = children.pop(0)
            # 跳过叶节点
            if not any((child._left, child._right)):
                continue
            children.append(child._left)
            children.append(child._right)
            # 以内部节点为单节点的预测误差
            c_t = np.sum((y - child._val) ** 2)
            T_t_results = []
            # 内部节点的叶结点个数
            leaf_nums = 0
            child_sub_node = [child._left, child._right]
            while child_sub_node:
                sub_node = child_sub_node.pop(0)
                if not any((sub_node._left, sub_node._right)):
                    leaf_nums += 2
                    continue
                child_sub_node.append(sub_node._left)
                child_sub_node.append(sub_node._right)
            # 求以内部节点为根节点的子树的预测误差
            for x in X:
                split_feature, split_point = child._best_pair
                node = child
                while split_feature != None:
                    if x[split_feature] <= split_point:
                        node = node._left
                    else:
                        node = node._right
                    split_feature, split_point = node._best_pair
                T_t_results.append(node._val)
            T_t_results = np.array(T_t_results)
            c_T_t = np.sum((y - T_t_results) ** 2)
            # 剪枝阈值
            g_t = (c_t - c_T_t) / (leaf_nums - 1)
            if g_t < alpha:
                alpha = g_t
                min_gt_tree = child
        # 将内部节点转换为叶节点
        min_gt_tree._left = None
        min_gt_tree._right = None
        min_gt_tree._best_pair = (None, None)
        return (alpha, root)
        
    def _poster_pruning(self, X: np.ndarray, y: np.ndarray):
        """
        对已经生成的决策树进行剪枝
        """
        sub_trees_series = [(np.inf, self.root)]
        pruned_tree = self.root
        # 构造最优子树序列
        while True:
            alpha, pruned_tree = self._get_pruned_tree(X, y, pruned_tree)
            sub_trees_series.append((alpha, pruned_tree))
            if not any((pruned_tree._left._best_pair[0], pruned_tree._right._best_pair[0])):
                break
        # 交叉验证获得最优子树
        


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