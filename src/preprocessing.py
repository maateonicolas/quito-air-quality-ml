import pandas as pd
import numpy as np


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normaliza nombres de columnas:
    - minúsculas
    - sin espacios
    - espacios reemplazados por _
    """
    df = df.copy()

    df.columns = (
        df.columns
        .str.lower()
        .str.strip()
        .str.replace(" ", "_")
    )

    return df


def clean_air_quality_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpia el dataset de calidad del aire.

    Pasos:
    1. Normaliza nombres de columnas.
    2. Convierte datetime.
    3. Elimina registros sin fecha.
    4. Convierte columnas numéricas.
    5. Reemplaza valores negativos por NaN.
    6. Elimina duplicados.
    """
    df = df.copy()

    df = normalize_column_names(df)

    if "datetime" not in df.columns:
        raise ValueError("El dataset debe tener una columna llamada 'datetime'.")

    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")

    # Eliminar filas sin fecha válida
    df = df.dropna(subset=["datetime"])

    numeric_cols = [
        "pm25", "pm10", "no2", "o3", "so2", "co",
        "temperature", "humidity", "aqi"
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df.loc[df[col] < 0, col] = np.nan

    df = df.drop_duplicates()

    return df