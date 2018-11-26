import numpy as np 

# 二叉树节点类
class Node:

    def __init__(self, val: np.ndarray):
        self.val = val
        self.left = None
        self.right = None

    def __repr__(self):
        return "<Node: %s>" % str(self.val)

# 构造平衡 kd 树
class KDTree:

    def __init__(self, k=1):
        self._k = k
        self._depth = 0

    def fit(self, X: np.ndarray):
        """
        X: dataset, shape = (n_samples, n_features)
        """
        _, features = X.shape
        self._k = self._k if self._k <= features else features
        return self._generate(X)
        
    def _generate(self, X: np.ndarray):
        samples, _ = X.shape
        if samples == 0:
            return None
        split_dimension = self._depth % self._k
        if samples == 1:
            median = 0
        else:
            if samples % 2 == 1:
                median = (samples - 1) // 2
            else:
                median = samples // 2
            X = X[X[:, split_dimension].argsort()]
        self._depth += 1
        root = Node(X[median])
        root.left = self._generate(X[:median])
        root.right = self._generate(X[median+1:])
        return root

if __name__ == "__main__":
    kdt = KDTree(2)
    test = np.array([[2, 3], [5, 4], [9, 6], [4, 7], [8, 1], [7, 2]])
    root = kdt.fit(test)
    stack = [root]
    while stack:
        node = stack.pop()
        if node.right:
            stack.append(node.right)
        if node.left:
            stack.append(node.left)

    