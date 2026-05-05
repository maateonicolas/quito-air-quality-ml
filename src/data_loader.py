import pandas as pd
from pathlib import Path


def load_csv(path: str | Path) -> pd.DataFrame:
    """
    Carga un archivo CSV y devuelve un DataFrame.

    Parameters
    ----------
    path : str | Path
        Ruta del archivo CSV.

    Returns
    -------
    pd.DataFrame
        Dataset cargado.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {path}")

    return pd.read_csv(path)