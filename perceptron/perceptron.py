import numpy as np 

class Perceptron0:

    def __init__(self, eta=1):
        """
        eta: learning rate
        """
        self._eta = eta
        self._w = None
        self._b = 0
        self._alpha = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        X.shape = (n_samples, n_features)
        y.shape = (n_samples, 1)
        algorithm: 感知机学习算法对偶形式的实现
        """
        rows, _ = X.shape
        self._alpha = np.zeros((rows, 1))
        gram = np.dot(X, X.T)
        flag = True
        while flag:
            update_times = 0
            for i in range(rows):
                error_condition = y[i, 0] * (np.multiply(self._alpha.T, np.multiply(y.T, gram[i, :])).sum() + self._b)
                if error_condition <= 0:
                    update_times = 1
                    self._alpha[i] += self._eta
                    self._b += y[i, 0]
                    break
            if update_times == 0:
                flag = False
        self._w = np.multiply(y, np.multiply(self._alpha, X)).sum(axis=0)
        return self

    def predict(self, X: np.ndarray):
        print(self._w.shape)
        y = np.dot(self._w, X.T) + self._b
        y_label = np.where(y>=0, 1, -1).squeeze()
        return y_label


class Perceptron1:
    
    def __init__(self, eta=1):
        self._eta = eta
        self._w = None
        self._b = 0

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        X.shape = (n_samples, n_features)
        y.shape = (n_samples, 1)
        algorithm: 感知机学习算法的实现
        """
        rows, columns = X.shape
        self._w = np.empty((1, columns))
        flag = True
        while flag:
            update_times = 0
            for i in range(rows):
                error_condition = y[i, 0] * (np.multiply(self._w, X[i, :]).sum() + self._b)
                if error_condition <= 0:
                    update_times = 1
                    self._w += self._eta * y[i, 0] * X[i, :]
                    self._b += self._eta * y[i, 0]
            if update_times == 0:
                flag = False
        return self

    def predict(self, X: np.ndarray):
        y = np.dot(self._w, X.T) + self._b
        y_label = np.where(y>=0, 1, -1).squeeze()
        return y_label

if __name__ == "__main__":
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    import matplotlib.pyplot as plt
    import seaborn as sns 

    iris = sns.load_dataset('iris')
    # sns.pairplot(iris, hue='species')
    # plt.show()

    iris_preprocessed = iris[(iris['species'] == 'setosa') | (iris['species'] == 'virginica')][['petal_length', 'petal_width', 'species']]
    

    X = iris_preprocessed[['petal_length', 'petal_width']].values
    y_label = iris_preprocessed['species'].values
    y = np.where(y_label=='setosa', 1, -1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    p = Perceptron0()
    p.fit(X_train, y_train.reshape(-1, 1))
    y_pre = p.predict(X_test)
    print(accuracy_score(y_test, y_pre))
