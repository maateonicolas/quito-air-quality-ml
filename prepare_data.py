from src.config import RAW_SAMPLE_FILE, PROCESSED_SAMPLE_FILE
from src.data_loader import load_csv
from src.preprocessing import clean_air_quality_data
from src.features import add_time_features, add_target


def main():
    print("Cargando datos crudos...")
    df = load_csv(RAW_SAMPLE_FILE)

    print("Limpiando datos...")
    df = clean_air_quality_data(df)

    print("Creando variables temporales...")
    df = add_time_features(df)

    print("Creando variable objetivo...")
    df = add_target(df)

    print("Guardando dataset procesado...")
    PROCESSED_SAMPLE_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_SAMPLE_FILE, index=False)

    print("Proceso completado.")
    print(f"Archivo guardado en: {PROCESSED_SAMPLE_FILE}")
    print()
    print("Distribución de clases:")
    print(df["target"].value_counts())


if __name__ == "__main__":
    main()