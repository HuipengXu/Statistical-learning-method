import numpy as np
from collections import Counter


class Node:
    """
    kd 树节点类
    """

    def __init__(self, val: np.ndarray, label: int, dim: int):
        self.val = val
        self.label = label
        self.dim = dim
        self.parent = None
        self.left = None
        self.right = None

    def __repr__(self):
        return "<Node: %s>" % str(self.val)


class KNearestNeighborsClassifier:
    """
    k 近邻
    """

    def __init__(self, K: int = 1):
        self._root = None
        self.K = K

    # 构造平衡 kd 树
    def _build_kd_tree(self, X: np.ndarray, y: np.ndarray, k: int, depth: int = 0):
        """
        k: dimension of dataset
        """
        samples, _ = X.shape
        if samples == 0:
            return None
        # 选择切分维度
        split_dimension = depth % k
        # 选择中位数
        if samples == 1:
            median = 0
        else:
            if samples % 2 == 1:
                median = (samples - 1) // 2
            else:
                median = samples // 2
            sorted_index = X[:, split_dimension].argsort()
            X = X[sorted_index]
            y = y[sorted_index]
        depth += 1
        root = Node(X[median], y[median], split_dimension)
        root.left = self._build_kd_tree(X[:median], y[:median], k, depth)
        if root.left:
            root.left.parent = root
        root.right = self._build_kd_tree(X[median + 1:], y[median + 1:], k, depth)
        if root.right:
            root.right.parent = root
        return root

    # 训练
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        X: dataset, shape = (n_samples, n_features)
        """
        _, features = X.shape
        self._root = self._build_kd_tree(X, y, features)
        return self

    # 搜索最近叶节点
    def nearest_leaf_search(self, root: Node, x: np.ndarray, dimension: int = 0):
        if not (root.left or root.right):
            return root
        features = x.shape[0]
        if x[dimension] <= root.val[dimension]:
            dimension = (dimension + 1) % features
            if not root.left:
                return self.nearest_leaf_search(root.right, x, dimension)
            return self.nearest_leaf_search(root.left, x, dimension)
        else:
            dimension = (dimension + 1) % features
            if not root.right:
                return self.nearest_leaf_search(root.left, x, dimension)
            return self.nearest_leaf_search(root.right, x, dimension)

    def k_nearest_neighbors(self, x: np.ndarray):
        node = self._root
        distance = np.linalg.norm(node.val - x)
        candidate_nodes = []
        k_min_distance = np.full((self.K,), distance)
        k_nearest_neighbors = np.full((self.K,), node)
        flag = True

        def update(child: Node, candidate_child: Node):
            nonlocal node, flag
            if candidate_child:
                # 参考 https://blog.csdn.net/dobests/article/details/48580899
                dim_dis = abs(x[dim] - node.val[dim])
                if dim_dis < k_min_distance.max():
                    candidate_nodes.append(candidate_child)
            if child:
                node = child
                distance = np.linalg.norm(node.val - x)
                max_arg = k_min_distance.argmax()
                if distance < k_min_distance[max_arg]:
                    k_min_distance[max_arg] = distance
                    k_nearest_neighbors[max_arg] = node
            elif candidate_nodes:
                node = candidate_nodes.pop()
            else:
                flag = False

        while flag:
            dim = node.dim
            if x[dim] <= node.val[dim]:
                update(node.left, node.right)
            else:
                update(node.right, node.left)
        return k_nearest_neighbors

    def predict(self, X: np.ndarray):
        """
        X: dataset, shape = (n_samples, n_features)
        """
        ret = []
        for x in X:
            k_nearest_neighbors = self.k_nearest_neighbors(x)
            labels = [node.label for node in k_nearest_neighbors]
            label_counter = Counter(labels)
            label, _ = label_counter.most_common(1)[0]
            ret.append(label)
        return np.array(ret)


if __name__ == "__main__":
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    from sklearn.neighbors import KNeighborsClassifier
    import time

    iris = load_iris()
    X = iris.data
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=4)
    nn = KNearestNeighborsClassifier(3)
    nn.fit(X_train, y_train)
    y_pre = nn.predict(X_test)
    y_t = nn.predict(X_train)
    s = time.time()
    print(accuracy_score(y_train, y_t))
    print(accuracy_score(y_test, y_pre))
    e = time.time()
    print('cost time: %.10f' % (e - s))
    print('-' * 20)
    s1 = time.time()
    knn = KNeighborsClassifier(n_neighbors=3).fit(X_train, y_train)
    y_t = knn.predict(X_train)
    print(accuracy_score(y_train, y_t))
    print(knn.score(X_test, y_test))
    e1 = time.time()
    print('cost time: %.10f' % (e1 - s1))
