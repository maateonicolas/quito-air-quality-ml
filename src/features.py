import pandas as pd


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea variables temporales a partir de la columna datetime.

    Nuevas variables:
    - hour
    - dayofweek
    - month
    - is_weekend
    """
    df = df.copy()

    if "datetime" not in df.columns:
        raise ValueError("El dataset debe tener la columna 'datetime'.")

    df["hour"] = df["datetime"].dt.hour
    df["dayofweek"] = df["datetime"].dt.dayofweek
    df["month"] = df["datetime"].dt.month
    df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)

    return df


def classify_aqi(aqi: float) -> str:
    """
    Clasifica el AQI en tres categorías simples.

    0 - 50: buena
    51 - 100: moderada
    101+: mala
    """
    if pd.isna(aqi):
        return None

    if aqi <= 50:
        return "buena"
    elif aqi <= 100:
        return "moderada"
    else:
        return "mala"


def add_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea la variable objetivo target a partir de AQI.
    """
    df = df.copy()

    if "aqi" not in df.columns:
        raise ValueError("El dataset debe tener una columna llamada 'aqi'.")

    df["target"] = df["aqi"].apply(classify_aqi)

    # Eliminar filas sin target
    df = df.dropna(subset=["target"])

    return df
def classify_pm25(pm25: float) -> str:
    """
    Clasifica la calidad del aire usando PM2.5 como referencia.

    pm25 <= 15: buena
    15 < pm25 <= 35: moderada
    pm25 > 35: mala
    """
    if pd.isna(pm25):
        return None

    if pm25 <= 15:
        return "buena"
    elif pm25 <= 35:
        return "moderada"
    else:
        return "mala"


def add_target_from_pm25(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea la variable objetivo target a partir de PM2.5.
    """
    df = df.copy()

    if "pm25" not in df.columns:
        raise ValueError("El dataset debe tener una columna llamada 'pm25'.")

    df["target"] = df["pm25"].apply(classify_pm25)

    df = df.dropna(subset=["target"])

    return df