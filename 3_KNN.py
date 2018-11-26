import numpy as np 

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
        self._depth = 0
        self._root = None

    # 构造平衡 kd 树
    def _generate(self, X: np.ndarray, y: np.ndarray, k: int):
        """
        k: dimension of dataset
        """
        samples, _ = X.shape
        if samples == 0:
            return None
        # 选择切分维度
        split_dimension = self._depth % k
        # 选择中位数
        if samples == 1:
            median = 0
        else:
            if samples % 2 == 1:
                median = (samples - 1) // 2
            else:
                median = samples // 2
            X = X[X[:, split_dimension].argsort()]
        self._depth += 1
        root = Node(X[median], y[median])
        root.left = self._generate(X[:median], y[:median], k)
        if root.left:
            root.left.parent = root
        root.right = self._generate(X[median+1:], y[:median], k)
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
        if x[dimension] < root.val[dimension]:
            if not root.left:
                return root
            dimension = (dimension + 1) % features
            return self.nearest_leaf_search(root.left, x, dimension)
        elif x[dimension] == root.val[dimension]:
            return root
        else:
            if not root.right:
                return root
            dimension = (dimension + 1) % features
            return self.nearest_leaf_search(root.right, x, dimension)
    
    # 搜索最近邻
    def nearest_neighbor_search(self, root: Node, x: np.ndarray):
        """
        root: KDTree 的根节点
        x: 待搜索的目标点
        """
        nearest_node = self.nearest_leaf_search(root, x) 
        current_node = nearest_node.parent
        min_distance = np.linalg.norm(x - nearest_node.val)
        while current_node:
            child = current_node.left if current_node.right == nearest_node else current_node.right
            for node in [child, current_node]:
                distance = np.linalg.norm(x - node.val)
                if distance < min_distance:
                    min_distance = distance
                    nearest_node = node
            current_node = current_node.parent
        return nearest_node
    
    def predict(self, X: np.ndarray):
        """
        X: dataset, shape = (n_samples, n_features)
        """
        ret = []
        for x in X:
            nearest_neighbor = self.nearest_leaf_search(self._root, x)
            ret.append(nearest_neighbor.label)
        return np.array(ret)
        

if __name__ == "__main__":
    test = np.array([[2, 3], [5, 4], [9, 6], [4, 7], [8, 1], [7, 2]])
    

    