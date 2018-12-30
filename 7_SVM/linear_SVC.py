# @Time    : 2018/12/23 14:08
# @Author  : Xu Huipeng
# @Blog    : https://brycexxx.github.io/

import numpy as np
from sklearn.preprocessing import LabelBinarizer
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

class SupportVectorClassifier:

    def __init__(self, kernel='rbf', sigma: float=1.0, tol: float=1e-3, C: float=1.0, max_iter: int=1000):
        self.kernel = kernel
        self.sigma = sigma
        self.tol = tol
        self.C = C
        self.max_iter = max_iter

    def _k_x_X(self, x: np.ndarray):
        inner_product = None
        if self.kernel == 'rbf':
            inner_product = 1.0 / np.exp(((self.X - x.reshape(1, self.n_features)) ** 2).sum(axis=1) / (2 * self.sigma))
        elif self.kernel == 'poly':
            pass
        elif self.kernel == 'linear':
            inner_product = np.dot(x, self.X.T)
        else:
            raise KeyError('no such kernel function')
        return inner_product

    def _k_xi_xj(self, xi, xj):
        k_xi_xj = None
        if self.kernel == 'rbf':
            k_xi_xj = 1.0 / np.exp(((xi - xj) ** 2).sum() / (2 * self.sigma))
        elif self.kernel == 'poly':
            pass
        elif self.kernel == 'linear':
            k_xi_xj = np.dot(xi, xj.T)
        else:
            raise KeyError('no such kernel function')
        return k_xi_xj

    def _gx(self, x: np.ndarray) -> float:
        gx = np.dot(self._k_x_X(x), self.y * self.alpha)[0] + self.b
        return gx

    def _Ei(self, i: int) -> float:
        return self._gx(self.X[i]) - self.y[i]

    def _select_alpha2(self, idx1: int, E1: float) -> int:
        max_deltaE = 0
        idx2 = -1
        non_bounds = np.nonzero((self.alpha > 0) * (self.alpha < self.C))[0]
        if len(non_bounds) > 0:
            iter_index = non_bounds
        else:
            iter_index = range(self.n_samples)
        for k in iter_index:
            if k == idx1: continue
            Ek = self.Ei[k]
            deltaE = abs(E1 - Ek)
            if (deltaE > max_deltaE):
                idx2 = k
                max_deltaE = deltaE
        return idx2

    def _inner_loop(self, idx1: int):
        E1 = self.Ei[idx1]
        alpha1 = self.alpha[idx1].copy()
        r1 = E1 * self.y[idx1]
        if ((r1 < -self.tol and alpha1 < self.C) or (r1 > self.tol and alpha1 > 0)):
            idx2 = self._select_alpha2(idx1, E1)
            E2 = self.Ei[idx2]
            alpha2 = self.alpha[idx2].copy()
            if self.y[idx1] == self.y[idx2]:
                L = max(.0, alpha2 + alpha1 - self.C)
                H = min(self.C, alpha2 + alpha1)
            else:
                L = max(.0, alpha2 - alpha1)
                H = min(self.C, self.C + alpha2 - alpha1)
            if L == H: print('L == H'); return 0
            K_11 = self._k_xi_xj(self.X[idx1], self.X[idx1])
            K_22 = self._k_xi_xj(self.X[idx2], self.X[idx2])
            K_12 = self._k_xi_xj(self.X[idx1], self.X[idx2])
            eta = K_11 + K_22 - 2 * K_12
            s = self.y[idx1] * self.y[idx2]
            # 更新后的 alpha2
            if eta <= 0:
                f1 = self.y[idx1] * (E1 + self.b) - alpha1 * K_11 - s * alpha2 * K_12
                f2 = self.y[idx2] * (E2 + self.b) - s * alpha1 * K_12 - alpha2 * K_22
                L1 = alpha1 + s * (alpha2 - L)
                H1 = alpha1 + s * (alpha2 - H)
                phi_l = L1 * f1 + L * f2 + 0.5 * L1 ** 2 * K_11 + 0.5 * L ** 2 * K_22 + s * L * L1 * K_12
                phi_h = H1 * f1 + H * f2 + 0.5 * H1 ** 2 * K_11 + 0.5 * H ** 2 * K_22 + s * H * H1 * K_12
                if (phi_l < (phi_h - 1e-5)):
                    alpha2_new = L
                elif (phi_l > (phi_h + 1e-5)):
                    alpha2_new = H
                else:
                    alpha2_new = alpha2
            else:
                alpha2_new_unc = alpha2 + self.y[idx2] * (E1 - E2) / eta
                if alpha2_new_unc > H: alpha2_new = H
                elif alpha2_new_unc < L: alpha2_new = L
                else: alpha2_new = alpha2_new_unc
            if abs(alpha2_new - alpha2) < 1e-5: print('alpha2 moving not enough'); return 0
            # 更新后的 alpha1
            alpha1_new = alpha1 + s * (alpha2 - alpha2_new)
            # 更新 alpha1 alpha2
            self.alpha[[idx1, idx2]] = np.array([[alpha1_new, alpha2_new]]).T
            b1_new = - E1 - self.y[idx1] * K_11 * (alpha1_new - alpha1) - self.y[idx2] * \
                     K_12 * (alpha2_new - alpha2) + self.b
            b2_new = - E2 - self.y[idx1] * K_12 * (alpha1_new - alpha1) - self.y[idx2] * \
                     K_22 * (alpha2_new - alpha2) + self.b
            # 更新 self.b
            if 0 < alpha1_new < self.C: self.b = b1_new
            elif 0 < alpha2_new < self.C: self.b = b2_new
            else: self.b = (b1_new + b2_new) / 2.0
            # 更新 Ei
            for i in range(self.n_samples): self.Ei[i] = self._Ei(i)
            return 1
        else: return 0

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.X, self.y = X, y
        self.b = .0
        lb = LabelBinarizer(neg_label=-1)
        self.y = lb.fit_transform(self.y)
        self.classes_ = lb.classes_
        self.n_samples, self.n_features = self.X.shape
        self.alpha = np.zeros((self.n_samples, 1))
        self.Ei = np.zeros_like(self.alpha)
        for i in range(self.n_samples): self.Ei[i] = self._Ei(i)
        iter_ = 0; entire_set = True; alpha_pairs_changed = 0
        while (iter_ < self.max_iter) and (entire_set or alpha_pairs_changed > 0):
            alpha_pairs_changed = 0
            if not entire_set:
                non_bounds = np.nonzero((self.alpha > 0) * (self.alpha < self.C))[0]
                for i in non_bounds:
                    alpha_pairs_changed += self._inner_loop(i)
                    print("non-bound, iter: %d i:%d, pairs changed %d" % \
                    (iter_, i, alpha_pairs_changed))
                iter_ += 1
            else:
                for i in range(self.n_samples):
                    alpha_pairs_changed += self._inner_loop(i)
                    print("fullSet, iter: %d i:%d, pairs changed %d" % \
                    (iter_, i, alpha_pairs_changed))
                iter_ += 1
            if entire_set: entire_set = False
            elif alpha_pairs_changed == 0: entire_set = True
        return self

    def predict(self, X: np.ndarray):
        gx = np.array([self._gx(x) for x in X])
        ret = np.where(gx>=0, self.classes_[-1], self.classes_[0])
        return ret.squeeze()

    def plot_2d_SVM(self):
        pos_idx = (self.y == 1).squeeze()
        neg_idx = (self.y == -1).squeeze()
        sv_idx = (self.alpha != 0).squeeze()
        print(self.X[sv_idx])

        w = np.dot(self.X.T, np.multiply(self.alpha, self.y))

        plt.figure()
        plt.scatter(self.X[pos_idx][:, 0], self.X[pos_idx][:, 1], c='red', label='pos')
        plt.scatter(self.X[neg_idx][:, 0], self.X[neg_idx][:, 1], c='green', label='neg')
        ax = plt.gca()
        for sv_x, sv_y in self.X[sv_idx]:
            circle = Circle((sv_x, sv_y), radius=0.05, facecolor='none', edgecolor=(0,0.8,0.8), linewidth=3, alpha=0.5)
            ax.add_artist(circle)
        x = np.arange(self.X[:, 0].min(), self.X[:, 0].max(), 0.1)
        y = - (w[0] * x + self.b) / w[1]
        # plt.plot(x, y.squeeze(), label='separate hyperplane')
        # plt.plot(self.X[sv_idx][:, 0], self.X[sv_idx][:, 1])
        plt.legend(frameon=False)
        plt.show()


if __name__ == "__main__":
    from sklearn.datasets import load_breast_cancer
    from sklearn.svm import SVC
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score


    # bc = load_breast_cancer()
    # X = bc.data
    # y = bc.target
    # # two_dim = PCA(n_components=2, random_state=0).fit_transform(X)
    # # pos_idx = y == 1
    # # neg_idx = y != 1
    # # plt.figure()
    # # plt.scatter(X[pos_idx][:, 0], X[pos_idx][:, 1], c='red', label='pos')
    # # plt.scatter(X[neg_idx][:, 0], X[neg_idx][:, 1], c='green', label='neg')
    # # plt.legend(frameon=False)
    # # plt.show()
    # # exit()
    # train_X, test_X, train_y, test_y = train_test_split(X, y, random_state=0)
    # # standard_scaler = StandardScaler().fit(train_X)
    # # train_X = standard_scaler.transform(train_X)
    # # test_X = standard_scaler.transform(test_X)
    # svc = SupportVectorClassifier(kernel='rbf', sigma=0.61, max_iter=10000, tol=1e-5, C=0.31)
    # svc.fit(train_X, train_y)
    # y_pred = svc.predict(test_X)
    # train_y_pred = svc.predict(train_X)
    # print("train set accuracy is %.4f" % accuracy_score(train_y, train_y_pred))
    # print("test set accuracy is %.4f" %accuracy_score(test_y, y_pred))
    # sk_svc = SVC(max_iter=80000, random_state=42, gamma=100, tol=1e-4, C=0.7).fit(train_X, train_y)
    # print("train set accuracy apply scikit-learn is %.4f" % sk_svc.score(train_X, train_y))
    # print("test set accuracy apply scikit-learn is %.4f" % sk_svc.score(test_X, test_y))
    # def loadDataSet(fileName):
    #     """
    #     加载数据集
    #     :param fileName:
    #     :return:
    #     """
    #     dataMat = []
    #     labelMat = []
    #     fr = open(fileName)
    #     for line in fr.readlines():
    #         lineArr = line.strip().split('\t')
    #         dataMat.append([float(lineArr[0]), float(lineArr[1])])
    #         labelMat.append(float(lineArr[2]))
    #     return dataMat, labelMat
    #
    #
    def loadImages(fileName):
        '''
        加载文件
        :param fileName:要加载的文件路径
        :return: 数据集和标签集
        '''
        # 存放数据及标记
        dataArr = []
        labelArr = []
        fr = open(fileName)
        for line in fr.readlines():
            curLine = line.strip().split(',')
            dataArr.append([int(num) / 255 for num in curLine[1:]])
            if int(curLine[0]) == 0:
                labelArr.append(1)
            else:
                labelArr.append(-1)
        # 返回数据集和标记
        return dataArr, labelArr
    x, y = loadImages(r'..\ttest\Mnist\mnist_train\mnist_train.csv')
    svc = SupportVectorClassifier(kernel='rbf', sigma=0.099, C=1, tol=0.0001, max_iter=10000)
    svc.fit(np.array(x), np.array(y))
    # print(svc.alpha)
    # svc.plot_2d_SVM()
    pred_train_y = svc.predict(np.array(x))
    print(pred_train_y)
    test_x, test_y = loadImages(r'..\ttest\Mnist\mnist_test\mnist_test.csv')
    pred_test_y = svc.predict(np.array(test_x))
    print(accuracy_score(np.array(y), pred_train_y))
    print(accuracy_score(np.array(test_y), pred_test_y))






