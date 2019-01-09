import numpy as np 
from collections import Counter

class Node:
    """
    kd 树节点类
    """
    def __init__(self, val: np.ndarray, label):
        self.val = val
        self.label = label
        self.parent = None
        self.left = None
        self.right = None

    def __repr__(self):
        return "<Node: %s>" % str(self.val)

class KNearestNeighborsClassifier:
    """
    最近邻
    """
    def __init__(self, K: int=1):
        self._root = None
        self.K = K

    # 构造平衡 kd 树
    def _generate(self, X: np.ndarray, y: np.ndarray, k: int, depth: int=0):
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
        root = Node(X[median], y[median])
        root.left = self._generate(X[:median], y[:median], k, depth)
        if root.left:
            root.left.parent = root
        root.right = self._generate(X[median+1:], y[median+1:], k, depth)
        if root.right:
            root.right.parent = root
        return root

    # 训练
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        X: dataset, shape = (n_samples, n_features)
        """
        _, features = X.shape
        self._root = self._generate(X, y, features)
        return self

    # 搜索最近叶节点
    def nearest_leaf_search(self, root: Node, x: np.ndarray, dimension: int=0):
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
    
    # 搜索 k 近邻
    def nearest_neighbor_search(self, x: np.ndarray):
        """
        root: KDTree 的根节点
        x: 待搜索的目标点
        """
        nearest_node = self.nearest_leaf_search(self._root, x) 
        k_nearest_neighbors = [nearest_node] * self.K
        current_node = nearest_node.parent
        min_distance = np.linalg.norm(x - nearest_node.val)
        k_min_distance = [min_distance] * self.K
        child = nearest_node
        while current_node:
            other_child = current_node.left if current_node.right == child else current_node.right
            for node in [other_child, current_node]:
                if node:
                    # 如果新的节点与 x 的距离小于 k 个距离中最小的，则更新 k 近邻
                    distance = np.linalg.norm(x - node.val)
                    max_distance_of_k = max(k_min_distance)
                    max_index = k_min_distance.index(max_distance_of_k)
                    if distance < max_distance_of_k:
                        k_nearest_neighbors.pop(max_index)
                        k_min_distance.pop(max_index)
                        k_nearest_neighbors.append(node)
                        k_min_distance.append(distance)
            child = current_node
            current_node = current_node.parent
        return k_nearest_neighbors
    
    def predict(self, X: np.ndarray):
        """
        X: dataset, shape = (n_samples, n_features)
        """
        ret = []
        for x in X:
            k_nearest_neighbors = self.nearest_neighbor_search(x)
            labels = [node.label for node in k_nearest_neighbors]
            label_counter = Counter(labels)
            label, _ = label_counter.most_common(1)[0]
            ret.append(label)
        return np.array(ret)
        

if __name__ == "__main__":
    test = np.array([[2, 3], [5, 4], [9, 6], [4, 7], [8, 1], [7, 2]])
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    from sklearn.neighbors import KNeighborsClassifier, KDTree

    iris = load_iris()
    X = iris.data
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=4)
    nn = KNearestNeighborsClassifier(5)
    nn.fit(X_train, y_train)
    y_pre = nn.predict(X_test)
    y_t = nn.predict(X_train)
    print(accuracy_score(y_train, y_t))
    print(accuracy_score(y_test, y_pre))
    # knn = KNeighborsClassifier(n_neighbors=2).fit(X_train, y_train)
    # y_t = knn.predict(X_train)
    # print(accuracy_score(y_train, y_t))
    # print(knn.score(X_test, y_test))





    