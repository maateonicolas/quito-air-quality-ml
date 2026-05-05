import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


def split_features_target(df: pd.DataFrame):
    """
    Separa variables predictoras X y variable objetivo y.

    Esta función se usa para el dataset sintético.
    """

    feature_cols = [
        "pm25", "pm10", "no2", "o3", "so2", "co",
        "temperature", "humidity",
        "hour", "dayofweek", "month", "is_weekend",
        "station"
    ]

    feature_cols = [col for col in feature_cols if col in df.columns]

    X = df[feature_cols]
    y = df["target"]

    return X, y


def split_features_target_real(df: pd.DataFrame):
    """
    Separa X e y para datos reales.

    Importante:
    No usamos pm25 como predictor porque el target fue construido desde pm25.
    Esto evita data leakage.
    """

    feature_cols = [
        "pm10", "no2", "o3", "so2", "co",
        "temperature", "humidity", "precipitation", "wind_speed",
        "hour", "dayofweek", "month", "is_weekend",
        "station"
    ]

    feature_cols = [col for col in feature_cols if col in df.columns]

    X = df[feature_cols]
    y = df["target"]

    return X, y


def create_train_test_split(X, y, test_size=0.3, random_state=42):
    """
    Divide los datos en entrenamiento y prueba.

    stratify=y mantiene la proporción de clases en train y test.
    """
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )


def build_preprocessor(X: pd.DataFrame):
    """
    Crea el preprocesador:
    - Imputa valores faltantes numéricos con la mediana.
    - Escala variables numéricas.
    - Imputa variables categóricas con el valor más frecuente.
    - Codifica variables categóricas con OneHotEncoder.
    """

    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )

    return preprocessor