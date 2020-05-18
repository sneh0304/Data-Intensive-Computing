# -*- coding: utf-8 -*-
"""
Predicitve_Analytics.py
"""
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

#method for normalizing dataset
#start:
def normalize(x):
    x_norm = np.zeros(x.shape)
    mean = np.mean(x, axis = 0)
    sd = np.std(x, axis = 0)
    x_norm = (x-mean)/sd
    return x_norm
#end

#Reference taken from https://towardsdatascience.com/random-forests-and-decision-trees-from-scratch-in-python-3e4fa5ae4249
class DecisionTree:
    def __init__(self, x, y, feature_indices, depth):
        self.x = x
        self.y = y
        self.feature_indices = feature_indices
        self.depth = depth
        self.val = int(np.mean(y))
        self.gini = float('inf')
        self.split()
    
    def split(self):
        for i in self.feature_indices:
            self.get_split(i)
        
        if self.gini == float('inf') or self.depth == 0:
            return
        l_feature_indices = np.random.permutation(self.x.shape[1])[:len(self.feature_indices)]
        r_feature_indices = np.random.permutation(self.x.shape[1])[:len(self.feature_indices)]
        self.left = DecisionTree(self.x[self.l_node], self.y[self.l_node], l_feature_indices, self.depth - 1)
        self.right = DecisionTree(self.x[self.r_node], self.y[self.r_node], r_feature_indices, self.depth - 1)

    def get_split(self, i):
        if len(set(self.y)) == 1:
            return
        x = self.x[:, i]
        sorted_x = np.sort(x)
        for j in range(sorted_x.shape[0] - 1):
            split = (sorted_x[j] + sorted_x[j + 1]) / 2
            l_node = np.nonzero(x <= split)[0]
            r_node = np.nonzero(x > split)[0]
            gini = self.calculate_gini(l_node, r_node)
            if gini < self.gini:
                self.gini, self.split = gini, split
                self.l_node, self.r_node = l_node, r_node
                self.split_index = i
    
    def calculate_gini(self, l, r):
        n = len(l) + len(r)
        gini = 0.0
        for child in [l, r]:
            size = len(child)
            if size == 0:
                continue
            _class, count = np.unique(self.y[child], return_counts = True)
            count = count / size
            score = np.sum(np.multiply(count, count))
            gini += (1.0 - score) * (size / n)
        return gini

    def predict(self, x):
        return np.array([self._predict(xi) for xi in x])

    def _predict(self, xi):
        if self.gini == float('inf') or self.depth == 0:
            return self.val
        t = self.left if xi[self.split_index] <= self.split else self.right
        return t._predict(xi)
    
def ConfusionMatrix(y_true,y_pred):
    
    """
    :type y_true: numpy.ndarray
    :type y_pred: numpy.ndarray
    :rtype: float
    """  
    n = np.unique(y_true).shape[0]
    temp = y_true * n
    temp = temp + y_pred
    return np.histogram(temp, n * n)[0].reshape(n, n)

def Accuracy(y_true,y_pred):
    """
    :type y_true: numpy.ndarray
    :type y_pred: numpy.ndarray
    :rtype: float
    
    """
    return np.mean(y_true == y_pred)

def Recall(y_true,y_pred):
    """
    :type y_true: numpy.ndarray
    :type y_pred: numpy.ndarray
    :rtype: float
    """
    cm = ConfusionMatrix(y_true, y_pred)
    return np.diag(cm) / np.sum(cm, axis = 1)

def Precision(y_true,y_pred):
    """
    :type y_true: numpy.ndarray
    :type y_pred: numpy.ndarray
    :rtype: float
    """
    cm = ConfusionMatrix(y_true, y_pred)
    return np.diag(cm) / np.sum(cm, axis = 0)
    
def WCSS(Clusters):
    """
    :Clusters List[numpy.ndarray]
    :rtype: float
    """
    centroids = np.zeros((Clusters[0].shape[1], len(Clusters)))
    for i, cluster in enumerate(Clusters):
        centroids[:, i] = np.mean(cluster, axis = 0)
    wcss = 0
    for i, cluster in enumerate(Clusters):
        x = np.tile(centroids[:, i], (cluster.shape[0], 1))
        wcss = np.linalg.norm(cluster - x)
    return wcss

def KNN(X_train,Y_train,X_test,Y_test,norm = False):
    """
    :type X_train: numpy.ndarray
    :type X_test: numpy.ndarray
    :type Y_train: numpy.ndarray
    
    :rtype: numpy.ndarray
    """
    if not isinstance(X_train, np.ndarray):
        X_train = np.array(X_train)
    if not isinstance(Y_train, np.ndarray):
        Y_train = np.array(Y_train)
    if not isinstance(X_test, np.ndarray):
        X_test = np.array(X_test)
    if not isinstance(Y_test, np.ndarray):
        Y_test = np.array(Y_test)
        
    if norm:
        X_train = normalize(X_train)
        X_test = normalize(X_test)
    
    k = 5
    res = []
    for x in X_test:
        x = np.tile(x, (X_train.shape[0], 1))
        distances = np.linalg.norm(x - X_train, axis = 1)
        k_idx = np.argsort(distances)[:k]
        k_neighbor_labels = Y_train[k_idx]
        most_common, count = np.unique(k_neighbor_labels, return_counts = True)
        res.append(most_common[np.argmax(count)])
    
    print('KNN Accuracy: ' + str(Accuracy(Y_test, np.array(res))))
    return np.array(res)
    
def RandomForest(X_train,Y_train,X_test,Y_test,norm = False):
    """
    :type X_train: numpy.ndarray
    :type X_test: numpy.ndarray
    :type Y_train: numpy.ndarray
    
    :rtype: numpy.ndarray
    """
    if not isinstance(X_train, np.ndarray):
        X_train = np.array(X_train)
    if not isinstance(Y_train, np.ndarray):
        Y_train = np.array(Y_train)
    if not isinstance(X_test, np.ndarray):
        X_test = np.array(X_test)
    if not isinstance(Y_test, np.ndarray):
        Y_test = np.array(Y_test)
        
    if norm:
        X_train = normalize(X_train)
        X_test = normalize(X_test)
    
    np.random.seed(0)
    total_samples, total_features = X_train.shape
    p = int(np.round(np.sqrt(total_features)))
    samples = int(0.63 * total_samples)
    
    trees = []
    for i in range(10):
        R_indices = np.random.permutation(total_samples)[:samples]
        C_indices = np.random.permutation(total_features)[:p]
        t = DecisionTree(X_train[R_indices], Y_train[R_indices], C_indices, 10)
        trees.append(t)
    pred = np.round([t.predict(X_test) for t in trees])
    
    pred_y = []
    for x in pred.T:
        pred_y.append(max(x, key = list(x).count))
    
    print('Random Forest Accuracy: ' + str(Accuracy(Y_test, np.array(pred_y))))
    return np.array(pred_y)
    
def PCA(X_train,N):
    """
    :type X_train: numpy.ndarray
    :type N: int
    :rtype: numpy.ndarray
    """
    
    mean = np.mean(X_train, axis = 0)
    X_train = X_train - mean
    cov = np.cov(X_train.T)
    eigenvalues, eigenvectors = np.linalg.eig(cov)
    eigenvectors = eigenvectors.T
    idxs = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idxs]
    eigenvectors = eigenvectors[idxs]
    components = eigenvectors[:N]
    
    return np.dot(X_train, components.T)
    
def Kmeans(X_train,N,norm = False):
    """
    :type X_train: numpy.ndarray
    :type N: int
    :rtype: List[numpy.ndarray]
    """
    if not isinstance(X_train, np.ndarray):
        X_train = np.array(X_train)
        
    if norm:
        X_train = normalize(X_train)
    
    examples = X_train.shape[0]
    features = X_train.shape[1]
    centroids = np.random.rand(features, N)
    
    while True:
        Euc_distance = []
        for i in range(N):
            x = np.tile(centroids[:, i], (X_train.shape[0], 1))
            distance = np.linalg.norm(X_train - x, axis = 1)
            Euc_distance.append(distance)
        Euc_distance = np.array(Euc_distance).T
        
        centroid_index = np.argmin(Euc_distance, axis = 1) + 1
    
        clusters = {}
        for i in range(N):
            clusters[i + 1] = np.array([]).reshape(0, features)
            
        for i in range(examples):
            clusters[centroid_index[i]] = np.r_['0, 2', clusters[centroid_index[i]], X_train[i]]
        
        old_centroids = centroids.copy()
        for i in range(N):
            centroids[:, i] = np.mean(clusters[i + 1], axis = 0)
        
        _sum = np.sum((old_centroids - centroids), axis = 0)
        if np.sum(_sum) == 0:
            break;
    
    return list(clusters.values())
        
        
def getEstimators():
    res = {}
    res['SVM'] = LinearSVC(multi_class = 'crammer_singer', random_state = 0, verbose = 1, max_iter = 500)
    res['LogR'] = LogisticRegression(solver = 'saga', multi_class = 'multinomial', random_state = 0, verbose = 1, max_iter = 1000)
    res['DT'] = DecisionTreeClassifier(criterion = 'gini', splitter = 'best', max_features = 'sqrt', random_state = 0)
    res['KNN'] = KNeighborsClassifier(n_neighbors = 5, algorithm = 'auto', weights = 'distance')
    
    return res

#Reference taken from https://stackoverflow.com/questions/20998083/show-the-values-in-the-grid-using-matplotlib
def visualizeCM(test_y, pred_y):
    cm1 = ConfusionMatrix(test_y, pred_y[0])
    cm2 = ConfusionMatrix(test_y, pred_y[1])
    cm3 = ConfusionMatrix(test_y, pred_y[2])
    cm4 = ConfusionMatrix(test_y, pred_y[3])
    
    fig, axs = plt.subplots(2, 2, figsize=(12,12))
    
    axs[0, 0].matshow(cm1, cmap = plt.cm.PuBuGn)
    axs[0, 0].set_title('CM for SVM')
    for (i, j), val in np.ndenumerate(cm1):
        axs[0, 0].text(j, i, str(val), va='center', ha='center')
    
    axs[0, 1].matshow(cm2, cmap = plt.cm.PuBuGn)
    axs[0, 1].set_title('CM for Logistic Regression')
    for (i, j), val in np.ndenumerate(cm2):
        axs[0, 1].text(j, i, str(val), va='center', ha='center')
    
    axs[1, 0].matshow(cm3, cmap = plt.cm.PuBuGn)
    axs[1, 0].set_title('CM for Decision Tree')
    for (i, j), val in np.ndenumerate(cm3):
        axs[1, 0].text(j, i, str(val), va='center', ha='center')
    
    axs[1, 1].matshow(cm4, cmap = plt.cm.PuBuGn)
    axs[1, 1].set_title('CM for KNN')
    for (i, j), val in np.ndenumerate(cm4):
        axs[1, 1].text(j, i, str(val), va='center', ha='center')
    
    for ax in axs.flat:
        ax.set(xlabel='Predicted label', ylabel='True label')
    
    for ax in axs.flat:
        ax.label_outer()
    
    fig.savefig('Confusion_Matrices.png')
    fig.show()

def visualizeGridParams(svm_results, dt_results, knn_results):
    plt.figure(1, figsize=(10, 6))
    plt.plot(np.linspace(1, 100, num = 11), svm_results['mean_train_score'])
    plt.plot(np.linspace(1, 100, num = 11), svm_results['mean_test_score'])
    plt.title('Grid Search SVM: Accuracy vs C')
    plt.ylabel('Accuracy')
    plt.xlabel('C')
    plt.legend(['Train', 'Validation'], loc='best')
    plt.savefig("grid_svm.png")
    plt.show()
    
    plt.figure(2, figsize=(10, 6))
    plt.plot([2, 5, 10, 20, None], dt_results['mean_train_score'])
    plt.plot([2, 5, 10, 20, None], dt_results['mean_test_score'])
    plt.title('Grid Search Decicion Tree: Accuracy vs max_depth')
    plt.ylabel('Accuracy')
    plt.xlabel('max_depth')
    plt.legend(['Train', 'Validation'], loc='best')
    plt.savefig("grid_DT.png")
    plt.show()
    
    plt.figure(3, figsize=(10, 6))
    plt.plot([2, 5, 10, 15, 20, 25, 30], knn_results['mean_train_score'])
    plt.plot([2, 5, 10, 15, 20, 25, 30], knn_results['mean_test_score'])
    plt.title('Grid Search KNN: Accuracy vs K')
    plt.ylabel('Accuracy')
    plt.xlabel('K')
    plt.legend(['Train', 'Validation'], loc='best')
    plt.savefig("grid_KNN.png")
    plt.show()

def SklearnSupervisedLearning(X_train,Y_train,X_test,Y_test,norm = False,gridSearch = False):
    """
    :type X_train: numpy.ndarray
    :type X_test: numpy.ndarray
    :type Y_train: numpy.ndarray
    
    :rtype: List[numpy.ndarray] 
    """
    if not isinstance(X_train, np.ndarray):
        X_train = np.array(X_train)
    if not isinstance(Y_train, np.ndarray):
        Y_train = np.array(Y_train)
    if not isinstance(X_test, np.ndarray):
        X_test = np.array(X_test)
    if not isinstance(Y_test, np.ndarray):
        Y_test = np.array(Y_test)
        
    if norm:
        X_train = normalize(X_train)
        X_test = normalize(X_test)
    
    estimators = getEstimators()
    res = []
    if gridSearch:
        svm = GridSearchCV(estimators['SVM'], param_grid = {'C' : np.linspace(1, 100, num = 11)}, scoring = 'accuracy', refit = 'accuracy', verbose = 1)
    else:
        svm = estimators['SVM']
    svm.fit(X_train, Y_train)
    y_pred = svm.predict(X_test)
    res.append(y_pred)
    print('SVM Accuracy: ' + str(Accuracy(Y_test, y_pred)))
    
    lr = estimators['LogR']
    lr.fit(X_train, Y_train)
    y_pred = lr.predict(X_test)
    res.append(y_pred)
    print('LogisticRegression Accuracy: ' + str(Accuracy(Y_test, y_pred)))
    
    if gridSearch:
        dt = GridSearchCV(estimators['DT'], param_grid = {'max_depth' : [2, 5, 10, 20, None]}, scoring = 'accuracy', refit = 'accuracy', verbose = 1)
    else:
        dt = estimators['DT']
    dt.fit(X_train, Y_train)
    y_pred = dt.predict(X_test)
    res.append(y_pred)
    print('DecisionTreeClassifier Accuracy: ' + str(Accuracy(Y_test, y_pred)))
    
    if gridSearch:
        knn = GridSearchCV(estimators['KNN'], param_grid = {'n_neighbors' : [2, 5, 10, 15, 20, 25, 30]}, scoring = 'accuracy', refit = 'accuracy', verbose = 1)
    else:
        knn = estimators['KNN']
    knn.fit(X_train, Y_train)
    y_pred = knn.predict(X_test)
    res.append(y_pred)
    print('KNeighborsClassifier Accuracy: ' + str(Accuracy(Y_test, y_pred)))
    
    if gridSearch:
        visualizeGridParams(svm.cv_results_, dt.cv_results_, knn.cv_results_)  
    
    visualizeCM(Y_test, res)
    return res
    
def SklearnVotingClassifier(X_train,Y_train,X_test,Y_test,norm = False):
    
    """
    :type X_train: numpy.ndarray
    :type X_test: numpy.ndarray
    :type Y_train: numpy.ndarray
    
    :rtype: List[numpy.ndarray] 
    """
    if not isinstance(X_train, np.ndarray):
        X_train = np.array(X_train)
    if not isinstance(Y_train, np.ndarray):
        Y_train = np.array(Y_train)
    if not isinstance(X_test, np.ndarray):
        X_test = np.array(X_test)
    if not isinstance(Y_test, np.ndarray):
        Y_test = np.array(Y_test)
    
    if norm:
        X_train = normalize(X_train)
        X_test = normalize(X_test)
    
    ensemble_model = VotingClassifier(estimators = getEstimators().items(), voting='hard')
    ensemble_model.fit(X_train, Y_train)
    y_pred = ensemble_model.predict(X_test)
    print('VotingClassifier Accuracy: ' + str(Accuracy(Y_test, y_pred)))
    
    return y_pred
"""
Create your own custom functions for Matplotlib visualization of hyperparameter search. 
Make sure that plots are labeled and proper legends are used
"""