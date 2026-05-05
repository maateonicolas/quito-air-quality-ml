from src.config import PROCESSED_SAMPLE_FILE
from src.data_loader import load_csv
from src.train import (
    split_features_target,
    create_train_test_split,
    build_preprocessor
)
from src.models import (
    build_logistic_model,
    build_tree_model,
    build_random_forest_model
)
from src.evaluate import evaluate_classifier


def main():
    print("Cargando dataset procesado...")
    df = load_csv(PROCESSED_SAMPLE_FILE)

    print("Separando X e y...")
    X, y = split_features_target(df)

    print("Dividiendo train/test...")
    X_train, X_test, y_train, y_test = create_train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42
    )

    print()
    print("Tamaño de entrenamiento:", X_train.shape)
    print("Tamaño de prueba:", X_test.shape)
    print()
    print("Distribución en entrenamiento:")
    print(y_train.value_counts())
    print()
    print("Distribución en prueba:")
    print(y_test.value_counts())
    print()

    preprocessor = build_preprocessor(X_train)

    models = {
        "Logistic Regression": build_logistic_model(preprocessor),
        "Decision Tree": build_tree_model(preprocessor),
        "Random Forest": build_random_forest_model(preprocessor)
    }

    for name, model in models.items():
        print("=" * 70)
        print(name)
        print("=" * 70)

        model.fit(X_train, y_train)
        evaluate_classifier(model, X_test, y_test)
        print()


if __name__ == "__main__":
    main()