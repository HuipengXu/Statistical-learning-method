import numpy as np 

# TODO kd 树构造错误
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

class NearestNeighbor:
    """
    最近邻
    """
    def __init__(self):
        self._root = None

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
        print(depth)
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
    
    # 搜索最近邻
    def nearest_neighbor_search(self, x: np.ndarray):
        """
        root: KDTree 的根节点
        x: 待搜索的目标点
        """
        nearest_node = self.nearest_leaf_search(self._root, x) 
        current_node = nearest_node.parent
        min_distance = np.linalg.norm(x - nearest_node.val)
        child = nearest_node
        while current_node:
            other_child = current_node.left if current_node.right == child else current_node.right
            for node in [other_child, current_node]:
                if node:
                    distance = np.linalg.norm(x - node.val)
                    if distance < min_distance:
                        min_distance = distance
                        nearest_node = node
            child = current_node
            current_node = current_node.parent
        # print(min_distance)
        return nearest_node
    
    def predict(self, X: np.ndarray):
        """
        X: dataset, shape = (n_samples, n_features)
        """
        ret = []
        for x in X:
            nearest_neighbor = self.nearest_neighbor_search(x)
            ret.append(nearest_neighbor.label)
        return np.array(ret)
        

if __name__ == "__main__":
    test = np.array([[2, 3], [5, 4], [9, 6], [4, 7], [8, 1], [7, 2]])
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    from sklearn.neighbors import KNeighborsClassifier, KDTree

    iris = load_iris()
    nn = NearestNeighbor()
    X = iris.data
    y = iris.target
    # nn = NearestNeighbor().fit(test, np.array([4, 2, 3, 5, 6, 1]))
    # nn.predict(np.array([[7.5, 1.5]]))
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    nn.fit(X_train, y_train)
    y_pre = nn.predict(X_test)
    print(accuracy_score(y_test, y_pre))
    # knn = KNeighborsClassifier(n_neighbors=1, algorithm='kd_tree').fit(X_train, y_train)
    # y_ = knn.predict(X_test)
    # print(accuracy_score(y_test, y_))
    # print('----------------')
    # nn = NearestNeighbor().fit(X_train, y_train)
    # nn = NearestNeighbor().fit(X_train, y_train)
    # for x in X_test:
        # print(nn.nearest_neighbor_search(x))




    