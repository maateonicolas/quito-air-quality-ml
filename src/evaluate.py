from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def evaluate_classifier(model, X_test, y_test):
    """
    Evalúa un modelo de clasificación.
    """

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    print(f"Accuracy: {accuracy:.4f}")
    print()
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print()
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))