import pandas as pd
from pathlib import Path
from functools import reduce


RAW_DIR = Path("data/raw/real")
OUTPUT_FILE = Path("data/raw/quito_air_quality_real.csv")


FILES = {
    "MonthlyReport.csv": "pm25",
    "MonthlyReport (2).csv": "pm10",
    "MonthlyReport (3).csv": "o3",
    "MonthlyReport (4).csv": "no2",
    "MonthlyReport (5).csv": "co",
    "MonthlyReport (6).csv": "so2",
    "MonthlyReport (1).csv": "temperature",
    "MonthlyReport (7).csv": "humidity",
    "MonthlyReport (8).csv": "precipitation",
    "MonthlyReport (9).csv": "wind_speed",
}


MONTH_MAP = {
    "January": 1,
    "February": 2,
    "March": 3,
    "April": 4,
    "May": 5,
    "June": 6,
    "July": 7,
    "August": 8,
    "September": 9,
    "October": 10,
    "November": 11,
    "December": 12,
}


def extract_month_year(parameter_line: str):
    """
    Extrae mes y año desde una línea Parameter.
    Si no encuentra un mes real, devuelve None, None.
    """

    parts = [p.strip() for p in parameter_line.split(",")]

    month = None
    year = None

    for part in parts:
        if part in MONTH_MAP:
            month = MONTH_MAP[part]

        if part.isdigit() and len(part) == 4:
            year = int(part)

    return month, year


def parse_numeric_value(value_raw: str):
    """
    Convierte texto a número.
    Maneja vacíos y valores como '.60'.
    """

    value_raw = value_raw.strip()

    if value_raw == "":
        return None

    return pd.to_numeric(value_raw, errors="coerce")


def read_monthly_report(file_path: Path, variable_name: str) -> pd.DataFrame:
    """
    Convierte un reporte mensual del Municipio de Quito a formato largo.

    El archivo puede contener varios bloques. Cada bloque empieza con:
    Parameter: ... Month ... Year ...

    Solo se procesan bloques que tengan un mes válido.
    Al final se agrupa por datetime para evitar duplicados.
    """

    with open(file_path, encoding="latin1") as file:
        lines = file.readlines()

    rows = []

    current_month = None
    current_year = None
    active_block = False

    for line in lines:
        line = line.strip()

        if not line:
            continue

        # Nuevo bloque
        if line.startswith("Parameter:"):
            current_month, current_year = extract_month_year(line)
            active_block = current_month is not None and current_year is not None
            continue

        # Ignorar encabezados
        if line.startswith("Site Name:") or line.startswith("Day"):
            continue

        if not active_block:
            continue

        parts = line.split(",")

        if len(parts) < 25:
            continue

        day_raw = parts[0].strip()

        if not day_raw.isdigit():
            continue

        day = int(day_raw)

        for hour in range(24):
            value = parse_numeric_value(parts[hour + 1])

            try:
                datetime_value = pd.Timestamp(
                    year=current_year,
                    month=current_month,
                    day=day,
                    hour=hour
                )
            except ValueError:
                continue

            rows.append({
                "datetime": datetime_value,
                variable_name: value
            })

    df = pd.DataFrame(rows)

    if df.empty:
        raise ValueError(f"No se pudieron extraer datos válidos de {file_path}")

    # Clave: evitar datetime duplicados antes del merge.
    # Si hay varios registros para la misma hora, se promedian.
    duplicates_before = df["datetime"].duplicated().sum()

    df = (
        df
        .groupby("datetime", as_index=False)[variable_name]
        .mean()
        .sort_values("datetime")
        .reset_index(drop=True)
    )

    print(f"Duplicados antes de agrupar en {variable_name}: {duplicates_before}")
    print(f"Filas después de agrupar {variable_name}: {df.shape[0]}")

    return df


def main():
    all_dfs = []

    for filename, variable_name in FILES.items():
        file_path = RAW_DIR / filename

        if not file_path.exists():
            raise FileNotFoundError(f"No se encontró: {file_path}")

        print("=" * 70)
        print(f"Procesando {filename} -> {variable_name}")

        df_variable = read_monthly_report(file_path, variable_name)

        print(df_variable.head())
        print("Shape:", df_variable.shape)
        print()

        all_dfs.append(df_variable)

    print("=" * 70)
    print("Uniendo variables por datetime...")

    df_final = reduce(
        lambda left, right: pd.merge(left, right, on="datetime", how="outer"),
        all_dfs
    )

    df_final = df_final.sort_values("datetime").reset_index(drop=True)

    df_final["station"] = "Quito"

    ordered_cols = [
        "datetime",
        "station",
        "pm25",
        "pm10",
        "no2",
        "o3",
        "so2",
        "co",
        "temperature",
        "humidity",
        "precipitation",
        "wind_speed",
    ]

    ordered_cols = [col for col in ordered_cols if col in df_final.columns]
    df_final = df_final[ordered_cols]

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df_final.to_csv(OUTPUT_FILE, index=False)

    print("Dataset real generado.")
    print(f"Archivo guardado en: {OUTPUT_FILE}")
    print()
    print(df_final.head())
    print()
    print("Shape final:", df_final.shape)
    print()
    print("Valores faltantes:")
    print(df_final.isna().sum())

    print()
    print("Rango de fechas:")
    print(df_final["datetime"].min(), "→", df_final["datetime"].max())


if __name__ == "__main__":
    main()