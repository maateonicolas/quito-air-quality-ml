from src.config import REAL_RAW_FILE, REAL_PROCESSED_FILE
from src.data_loader import load_csv
from src.preprocessing import clean_air_quality_data
from src.features import add_time_features, add_target_from_pm25


def main():
    print("Cargando datos reales...")
    df = load_csv(REAL_RAW_FILE)

    print("Limpiando datos...")
    df = clean_air_quality_data(df)

    print("Creando variables temporales...")
    df = add_time_features(df)

    print("Creando target desde PM2.5...")
    df = add_target_from_pm25(df)

    print("Guardando dataset real procesado...")
    REAL_PROCESSED_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(REAL_PROCESSED_FILE, index=False)

    print("Proceso completado.")
    print(f"Archivo guardado en: {REAL_PROCESSED_FILE}")
    print()
    print("Shape:", df.shape)
    print()
    print("Distribución de clases:")
    print(df["target"].value_counts())
    print()
    print("Valores faltantes:")
    print(df.isna().sum())


if __name__ == "__main__":
    main()