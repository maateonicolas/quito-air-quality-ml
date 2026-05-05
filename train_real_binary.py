from src.config import REAL_PROCESSED_FILE
from src.data_loader import load_csv
from src.train import (
    split_features_target_real,
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
    print("Cargando dataset real procesado...")
    df = load_csv(REAL_PROCESSED_FILE)

    print("Convirtiendo target multiclase a binario...")
    df["target"] = df["target"].apply(
        lambda x: "buena" if x == "buena" else "no_buena"
    )

    print("Distribución binaria:")
    print(df["target"].value_counts())
    print()

    print("Separando X e y para datos reales...")
    X, y = split_features_target_real(df)

    print("Columnas usadas como predictores:")
    print(X.columns.tolist())
    print()

    print("Dividiendo train/test...")
    X_train, X_test, y_train, y_test = create_train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42
    )

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