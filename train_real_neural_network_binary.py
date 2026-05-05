import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

from src.config import REAL_PROCESSED_FILE
from src.data_loader import load_csv
from src.train import split_features_target_real, build_preprocessor


def build_binary_neural_network(input_dim: int):
    """
    Red neuronal para clasificación binaria.

    Salida:
    - 1 neurona
    - activación sigmoid
    """

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),

        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.2),

        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dropout(0.2),

        tf.keras.layers.Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model


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

    print("Codificando etiquetas...")
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    print("Clases:")
    for index, class_name in enumerate(label_encoder.classes_):
        print(f"{index}: {class_name}")
    print()

    print("Dividiendo train/test...")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=0.3,
        random_state=42,
        stratify=y_encoded
    )

    print("Tamaño de entrenamiento:", X_train.shape)
    print("Tamaño de prueba:", X_test.shape)
    print()

    print("Construyendo preprocesador...")
    preprocessor = build_preprocessor(X_train)

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    if hasattr(X_train_processed, "toarray"):
        X_train_processed = X_train_processed.toarray()
        X_test_processed = X_test_processed.toarray()

    print("Shape train procesado:", X_train_processed.shape)
    print("Shape test procesado:", X_test_processed.shape)
    print()

    print("Calculando pesos de clase...")
    class_weights_array = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_train),
        y=y_train
    )

    class_weights = {
        int(class_index): float(weight)
        for class_index, weight in zip(np.unique(y_train), class_weights_array)
    }

    print("Pesos de clase:", class_weights)
    print()

    input_dim = X_train_processed.shape[1]

    print("Construyendo red neuronal binaria...")
    model = build_binary_neural_network(input_dim)

    model.summary()

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True
    )

    print("Entrenando red neuronal...")
    history = model.fit(
        X_train_processed,
        y_train,
        validation_split=0.2,
        epochs=100,
        batch_size=32,
        callbacks=[early_stopping],
        class_weight=class_weights,
        verbose=1
    )

    print("Evaluando en test...")
    y_proba = model.predict(X_test_processed).ravel()

    # Umbral estándar para clasificación binaria
    y_pred = (y_proba >= 0.5).astype(int)

    accuracy = accuracy_score(y_test, y_pred)

    print()
    print(f"Test Accuracy: {accuracy:.4f}")
    print()

    print("Classification Report:")
    print(classification_report(
        y_test,
        y_pred,
        target_names=label_encoder.classes_,
        zero_division=0
    ))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    model.save("reports/real_binary_neural_network_model.keras")

    print()
    print("Modelo guardado en: reports/real_binary_neural_network_model.keras")


if __name__ == "__main__":
    main()