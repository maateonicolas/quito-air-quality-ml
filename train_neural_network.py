import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.utils.class_weight import compute_class_weight

from src.config import PROCESSED_SAMPLE_FILE
from src.data_loader import load_csv
from src.train import split_features_target, build_preprocessor


def build_neural_network(input_dim: int, num_classes: int):
    """
    Construye una red neuronal densa para clasificación multiclase.
    """

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),

        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.2),

        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dropout(0.2),

        tf.keras.layers.Dense(num_classes, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


def main():
    print("Cargando dataset procesado...")
    df = load_csv(PROCESSED_SAMPLE_FILE)

    print("Separando X e y...")
    X, y = split_features_target(df)

    print("Codificando etiquetas...")
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    print("Clases:")
    for index, class_name in enumerate(label_encoder.classes_):
        print(f"{index}: {class_name}")

    print("Dividiendo train/test...")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=0.3,
        random_state=42,
        stratify=y_encoded
    )

    print("Construyendo preprocesador...")
    preprocessor = build_preprocessor(X_train)

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # Algunas transformaciones devuelven matrices sparse.
    # TensorFlow necesita arrays densos.
    if hasattr(X_train_processed, "toarray"):
        X_train_processed = X_train_processed.toarray()
        X_test_processed = X_test_processed.toarray()

    print("Shape train procesado:", X_train_processed.shape)
    print("Shape test procesado:", X_test_processed.shape)

    input_dim = X_train_processed.shape[1]
    num_classes = len(np.unique(y_encoded))

    print("Calculando pesos de clase...")
    class_weights_array = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_train),
        y=y_train
    )

    class_weights = {
        class_index: weight
        for class_index, weight in zip(np.unique(y_train), class_weights_array)
    }

    print("Pesos de clase:", class_weights)

    print("Construyendo red neuronal...")
    model = build_neural_network(input_dim, num_classes)

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
    y_proba = model.predict(X_test_processed)
    y_pred = np.argmax(y_proba, axis=1)

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

    # Guardar modelo entrenado
    model.save("reports/neural_network_model.keras")
    print()
    print("Modelo guardado en: reports/neural_network_model.keras")


if __name__ == "__main__":
    main()