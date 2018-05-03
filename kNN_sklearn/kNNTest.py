import numpy as np
from kNNClf import KNNClassfier
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from model_selection import train_test_split
from preprocessing import StandardScaler
import matplotlib.pyplot as plt


def mykNN():
    data = datasets.load_digits()
    X = data.data
    y = data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, seed=5)

    """
    # best_p = -1
    best_score = 0.0
    best_k = -1
    best_method = ""
    for methode in ["uniform", "distance"]:
        for k in range(1, 11):
            # for p in range(1, 6):
            #     knn_clf = KNeighborsClassifier(n_neighbors=k, weights="distance", p=p)

            knn_clf = KNeighborsClassifier(n_neighbors=k, weights=methode)
            knn_clf.fit(X_train, y_train)
            score = knn_clf.score(X_test, y_test)
            if score > best_score:
                best_k = k
                best_score = score
                best_method = methode
                # best_p = p
        print("best_k=", best_k)
        print("best_score=", best_score)
        print("best_method=", best_method)
        # print("best_p=", best_p)
    """
    myKNN = KNNClassfier(3)
    standardScaler = StandardScaler()
    standardScaler.fit(X_train)
    X_standard_train = standardScaler.transform(X_train)
    X_standard_test = standardScaler.transform(X_test)
    # plt.scatter(X_standard_train[0, :], X_standard_train[1, :])
    # plt.show()
    myKNN.fit(X_standard_train, y_train)
    print(myKNN.score(X_standard_test, y_test))
    # plt.scatter(X_train[0, :], X_train[1, :])
    # plt.show()
    myKNN.fit(X_train, y_train)
    print(myKNN.score(X_test, y_test))


def sklearn_test():
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.model_selection import GridSearchCV

    data = datasets.load_digits()
    X = data.data
    y = data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    # param_grid = [
    #     {
    #         'weights': ['uniform'],
    #         'n_neighbors': [i for i in range(1, 11)]
    #     },
    #     {
    #         'weights': ['distance'],
    #         'n_neighbors': [i for i in range(1, 11)],
    #         'p': [i for i in range(1, 6)]
    #     }
    # ]
    # kNight = KNeighborsClassifier()
    # gride_search = GridSearchCV(kNight, param_grid, n_jobs=-1, verbose=2)
    # gride_search.fit(X_train, y_train)
    # print(gride_search.best_params_)
    kNight = KNeighborsClassifier(n_neighbors=4, p=4, weights='distance')
    standardScaler = StandardScaler()
    standardScaler.fit(X_train)
    X_standard_train = standardScaler.transform(X_train)
    X_standard_test = standardScaler.transform(X_test)
    # plt.scatter(X_standard_train[0, :], X_standard_train[1, :])
    # plt.show()
    kNight.fit(X_standard_train, y_train)
    print(kNight.score(X_standard_test, y_test))


if __name__ == "__main__":
    mykNN()
    # sklearn_test()
