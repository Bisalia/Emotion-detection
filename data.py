import numpy as np 
import matplotlib.pyplot as plt 
import cv2


def evaluate(A, Y, w):
    Yhat = np.argmax(A.dot(w), axis=1)
    return np.sum(Yhat == Y) / Y.shape[0]
def main():
    # load data
    with np.load('data/fer2013_train.npz') as data:
        X_train, Y_train = data['X'], data['Y']

    with np.load('data/fer2013_test.npz') as data:
        X_test, Y_test = data['X'], data['Y']
    
    # one-hot labels
    I = np.eye(6)
    Y_oh_train, Y_oh_test = I[Y_train], I[Y_test]
    d = 1000
    W = np.random.normal(size=(X_train.shape[1], d))
    # select first 100 dimensions
    A_train, A_test = X_train.dot(W), X_test.dot(W)
    # select first 100 dimensions
    #A_train, A_test = X_train[:, :100], X_test[:, :100]

    ...
    # train model
    I = np.eye(A_train.shape[1])
    w = np.linalg.inv(A_train.T.dot(A_train) + 1e10 * I).dot(A_train.T.dot(Y_oh_train))
    
    # evaluate model
    ols_train_accuracy = evaluate(A_train, Y_train, w)
    print('(ols) Train Accuracy:', ols_train_accuracy)
    ols_test_accuracy = evaluate(A_test, Y_test, w)
    print('(ols) Test Accuracy:', ols_test_accuracy)
    # representation with matplotlib
    labels = ['Train Accuracy','Test Accuracy']
    Accuracies = [ols_train_accuracy,ols_test_accuracy]
    plt.bar(labels, Accuracies)  # Création d'un graphique en barres
    plt.ylabel('Accuracy')  # Nommer l'axe y
    plt.title('OLS Accuracy Evaluation')  # Titre du graphique
    plt.ylim(0, 1)  # Limiter l'axe y de 0 à 1 (pourcentage)

    plt.show()  # Afficher le graphique

if __name__ == '__main__':
    main()
