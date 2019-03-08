# @Time    : 2019/1/3 9:25
# @Author  : Xu Huipeng
# @Blog    : https://brycexxx.github.io/
import numpy as np
from typing import Tuple
from sklearn.preprocessing import LabelBinarizer
from decision_tree.decision_tree_regressor import Node
from uuid import uuid1
import matplotlib.pyplot as plt


class BoostingStumpRegressor:
    """
    弱分类器是一个根节点直接连接
    两个叶节点的简单回归决策树
    """

    def __init__(self, n_estimators: int = 50):
        self.n_estimators = n_estimators
        self.tree_series = []
        self.n_features = 0

    def _split(self, x: np.ndarray, y: np.ndarray,
               split_feature_index: int, split_point: float) \
            -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        根据特征列和切分点将数据集分割为左右两部分
        """
        left_index = x[:, split_feature_index] <= split_point
        right_index = x[:, split_feature_index] > split_point
        return x[left_index, :], x[right_index, :], y[left_index], y[right_index]

    def _weak_regressor(self, X: np.ndarray, r: np.ndarray) -> tuple:
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
        # 残差
        # y = y.reshape(-1, 1)
        r = y.copy()
        for i in range(self.n_estimators):
            best_f_p = self._weak_regressor(X, r)
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


class BoostingStumpClassifier:
    """
    弱分类器为决策树桩
    """

    def __init__(self, n_estimators: int = 50):
        self.n_estimators = n_estimators
        self.n_samples = 0
        self.n_features = 0
        self.all_classifiers = []
        self.classes_ = None

    def _weak_classifier(self, X: np.ndarray, y: np.ndarray, D: np.ndarray):
        min_error = 1
        best_classifier = {}
        for f in range(self.n_features):
            unique_x = np.unique(X[:, f])
            for p in unique_x:
                for not_equal in ['lt', 'gt']:
                    G = np.ones((self.n_samples,))
                    if not_equal == 'lt':
                        G[X[:, f] <= p] = -1.0
                    else:
                        G[X[:, f] > p] = -1.0
                    error_idx = G != y
                    error_rate = D[error_idx].sum()
                    if error_rate < min_error:
                        min_error = error_rate
                        best_classifier['f'] = f
                        best_classifier['p'] = p
                        best_classifier['not_equal'] = not_equal
                        best_classifier['error_rate'] = min_error
        return best_classifier

    def fit(self, X: np.ndarray, y: np.ndarray):
        lb = LabelBinarizer(neg_label=-1)
        y = lb.fit_transform(y).squeeze()
        self.classes_ = lb.classes_
        self.n_samples, self.n_features = X.shape
        # 初始化权重 D
        D = np.ones((self.n_samples, 1)) / self.n_samples
        for i in range(self.n_estimators):
            clf = self._weak_classifier(X, y, D)
            # print('after training %d estimator, error rate is %.3f' % (i, clf['error_rate']))
            if clf['error_rate'] == .0:
                break
            alpha = 0.5 * np.log((1 - clf['error_rate']) / (clf['error_rate'] + 1e-9))
            # print('alpha: %.3f' % alpha)
            self.all_classifiers.append((clf, alpha))
            y_pred = np.where(self.predict(X) == 1, 1, -1)
            print('after training %d estimator, total error rate: %.4f' % (i, (y_pred != y).sum() / self.n_samples))
            # 更新权重 D
            for j in range(self.n_samples):
                if clf['not_equal'] == 'lt':
                    g = -1.0 if X[j, clf['f']] <= clf['p'] else 1.0
                else:
                    g = -1.0 if X[j, clf['f']] > clf['p'] else 1.0
                # print(y[j], g)
                # print('before: %.5f' % D[j])
                D[j] *= np.exp(-alpha * y[j] * g)
                # print('after: %.5f' % D[j])
            D /= D.sum()
        return self

    def decision_function(self, X: np.ndarray):
        m = X.shape[0]
        ret = np.zeros((m,))
        for i in range(m):
            for clf, alpha in self.all_classifiers:
                if clf['not_equal'] == 'lt':
                    g = -1.0 if X[i, clf['f']] <= clf['p'] else 1.0
                else:
                    g = -1.0 if X[i, clf['f']] > clf['p'] else 1.0
                ret[i] += alpha * g
        return ret

    def predict(self, X: np.ndarray):
        ret = self.decision_function(X)
        ret = np.where(ret >= 0, self.classes_[-1], self.classes_[0])
        return ret

    def roc_auc(self, y_true: np.ndarray, y_score: np.ndarray):
        unique_y = np.unique(y_score)
        threshold = [(unique_y[i] + unique_y[i + 1]) / 2.0 for i in range(np.size(unique_y) - 1)]
        tpr = []
        fpr = []
        for t in threshold:
            y_pred = np.where(y_score >= t, self.classes_[-1], self.classes_[0])
            pos = (y_true == self.classes_[-1])
            pos_pred = (y_pred == self.classes_[-1])
            tp = (pos & pos_pred).sum()
            fn = pos.sum() - tp
            fp = pos_pred.sum() - tp
            neg = (y_true == self.classes_[0])
            neg_pred = (y_pred == self.classes_[0])
            tn = (neg & neg_pred).sum()
            tpr.append(tp / (tp + fn))
            fpr.append(fp / (tn + fp))
        plt.figure()
        plt.plot(fpr, tpr, label='roc')
        plt.plot([0, 1], [0, 1], label='random guess')
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.xlabel("false positive rate")
        plt.ylabel("true positive rate")
        plt.title('ROC curve for AdaBoost')
        plt.legend(frameon=False)
        plt.show()


class BoostingDecisionTreeRegressor:
    """
    以回归决策树作为弱分类器
    """

    def __init__(self, n_estimators: int = 50, max_depth: int = 3):
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

    def _generate_regression_tree(self, X: np.ndarray, y: np.ndarray, depth: int = 0):
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
            split_point = [(unique_point[i] + unique_point[i + 1]) / 2.0 for i in range(np.size(unique_point) - 1)]
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

    def fit(self, X: np.ndarray, y: np.ndarray):
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
            while node._left is not None:
                if x[split_feature] <= split_point:
                    node = node._left
                else:
                    node = node._right
                split_feature, split_point = node._best_pair
            ret.append(node._val)
        return np.array(ret)

    def predict(self, X: np.ndarray):
        m = X.shape[0]
        y_pred = np.zeros((m,))
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

    from sklearn.datasets import load_boston, load_breast_cancer
    from sklearn.metrics import r2_score, accuracy_score
    from sklearn.model_selection import train_test_split, GridSearchCV
    from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor, GradientBoostingRegressor
    from sklearn.tree import DecisionTreeRegressor
    import pandas as pd

    boston = load_breast_cancer()
    X, y = boston.data, boston.target
    train_X, test_X, train_y, test_y = train_test_split(X, y, random_state=0)
    # bt = BoostingStumpRegressor(n_estimators=18).fit(train_X, train_y)
    # y_pred = bt.predict(test_X)
    # print("boosting simple tree,  r2-score on test set: %.2f" % r2_score(test_y, y_pred))
    # bdt = BoostingDecisionTreeRegressor(n_estimators=250, max_depth=2).fit(train_X, train_y)
    # print("boosting decision tree,  r2-score on test set: %.2f" % r2_score(test_y, bdt.predict(test_X)))
    # abr = AdaBoostRegressor(
    #     DecisionTreeRegressor(max_depth=3, min_samples_split=20, min_samples_leaf=5),
    #     loss='linear', n_estimators=80, learning_rate=0.5, random_state=0
    # ).fit(train_X, train_y)
    # print("AdaBoostRegressor of sklearn, r2-score on test set: %.2f" % abr.score(test_X, test_y))
    # # rf = RandomForestRegressor(n_estimators=35, max_depth=20, random_state=0).fit(train_X, train_y)
    # # print("random forest,  r2-score on test set: %.2f" % rf.score(test_X, test_y))
    # gbr = GradientBoostingRegressor(n_estimators=152, max_depth=2, min_samples_split=2, random_state=0,
    #                                 learning_rate=0.18).fit(train_X, train_y)
    # print("gradient boosting regressor,  r2-score on test set: %.2f" % gbr.score(test_X, test_y))
    # # rf = GradientBoostingRegressor(random_state=42)
    # # parameters = {'loss': ['lad', 'ls', 'huber'], 'n_estimators': [150, 152, 154], 'max_depth': [2, 3, 5], 'min_samples_split': [2, 4, 6], 'learning_rate': [0.17, 0.18, 0.19]}
    # # rgr = GridSearchCV(rf, parameters, cv=5)
    # # rgr.fit(train_X, train_y)
    # # print(rgr.cv_results_)
    # # print(rgr.best_estimator_)
    # # print(rgr.best_score_)
    # # print(rgr.best_params_)
    # bsc = BoostingStumpClassifier().fit(train_X, train_y)
    # print(
    #     'Boosting Stump Classifier, accuracy score on test set: %.3f' % accuracy_score(test_y, bsc.predict(test_X)))
    def loadDataSet(fileName):
        numFeat = len(open(fileName).readline().split('\t'))
        dataMat = []
        labelMat = []
        fr = open(fileName)
        for line in fr.readlines():
            lineArr = []
            curLine = line.strip().split('\t')
            for i in range(numFeat - 1):
                lineArr.append(float(curLine[i]))
            dataMat.append(lineArr)
            labelMat.append(float(curLine[-1]))
        return np.array(dataMat), np.array(labelMat)


    def data_processing(filename):
        data = pd.read_csv(filename, sep='\s+', header=None)
        for i in data.columns:
            idx = (data[i] == .0).tolist()
            data.iloc[idx, i] = data[i].mean()
        data_array = data.values
        return data_array[:, 0: -1], data_array[:, -1]


    X, y = loadDataSet('..\\ttest\\horseColicTraining2.txt')
    bsc = BoostingStumpClassifier(n_estimators=10).fit(X, y)
    y_score = bsc.decision_function(X)
    bsc.roc_auc(y, y_score)
    # X_test, y_test = loadDataSet('..\\ttest\\horseColicTest2.txt')
    # print("Boosting Stump Classifier, accuracy score on test set: %.3f" % accuracy_score(y_test, bsc.predict(X_test)))
