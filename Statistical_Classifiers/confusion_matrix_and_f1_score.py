from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.datasets import load_wine
from sklearn.svm import SVC


def main():
    dataset = load_wine()
    X, Y = dataset.data, dataset.target
    X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=0)
    svc = SVC(kernel='rbf', C=1).fit(X_train, y_train)
    y_pred = svc.predict(X_test)
    confusion = confusion_matrix(y_test, y_pred)
    print('Confusion Matrix\n')
    print(confusion)
    print('\nClassification Report\n')
    print(classification_report(y_test, y_pred, target_names=['Null-Pixels', 'Non-Forest', 'Forest']))
    pass


if __name__ == "__main__":
    main()
