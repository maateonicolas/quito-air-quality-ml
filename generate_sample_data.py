import numpy as np
import pandas as pd
from pathlib import Path


def generate_synthetic_air_quality_data(n_rows=1000, seed=42):
    np.random.seed(seed)

    stations = ["Centro", "Cotocollao", "Carapungo", "Tumbaco", "Los Chillos"]

    datetime = pd.date_range(
        start="2026-01-01 00:00:00",
        periods=n_rows,
        freq="h"
    )

    station = np.random.choice(stations, size=n_rows)

    hour = datetime.hour
    month = datetime.month

    # Patrón simple: más contaminación en horas pico
    rush_hour_effect = np.where(
        ((hour >= 7) & (hour <= 9)) | ((hour >= 17) & (hour <= 20)),
        1.5,
        1.0
    )

    # Base por estación
    station_effect = {
        "Centro": 1.4,
        "Cotocollao": 1.1,
        "Carapungo": 1.3,
        "Tumbaco": 0.8,
        "Los Chillos": 1.0
    }

    station_factor = np.array([station_effect[s] for s in station])

    pm25 = np.random.normal(22, 8, n_rows) * rush_hour_effect * station_factor
    pm10 = np.random.normal(45, 15, n_rows) * rush_hour_effect * station_factor
    no2 = np.random.normal(25, 8, n_rows) * rush_hour_effect * station_factor
    o3 = np.random.normal(28, 7, n_rows)
    so2 = np.random.normal(5, 2, n_rows)
    co = np.random.normal(0.6, 0.2, n_rows) * rush_hour_effect

    temperature = np.random.normal(18, 3, n_rows)
    humidity = np.random.normal(65, 12, n_rows)

    # Evitar valores negativos
    pm25 = np.clip(pm25, 1, None)
    pm10 = np.clip(pm10, 1, None)
    no2 = np.clip(no2, 1, None)
    o3 = np.clip(o3, 1, None)
    so2 = np.clip(so2, 0.1, None)
    co = np.clip(co, 0.1, None)
    humidity = np.clip(humidity, 20, 100)

    # AQI sintético aproximado
    aqi = (
        pm25 * 1.4 +
        pm10 * 0.35 +
        no2 * 0.6 +
        o3 * 0.25 +
        co * 8
    )

    # Ruido
    aqi += np.random.normal(0, 8, n_rows)

    aqi = np.clip(aqi, 1, 200)

    df = pd.DataFrame({
        "datetime": datetime,
        "station": station,
        "pm25": pm25.round(2),
        "pm10": pm10.round(2),
        "no2": no2.round(2),
        "o3": o3.round(2),
        "so2": so2.round(2),
        "co": co.round(2),
        "temperature": temperature.round(2),
        "humidity": humidity.round(2),
        "aqi": aqi.round(2)
    })

    return df


def main():
    output_path = Path("data/raw/quito_air_quality_sample.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = generate_synthetic_air_quality_data(n_rows=1000)

    df.to_csv(output_path, index=False)

    print(f"Dataset sintético guardado en: {output_path}")
    print(df.head())
    print()
    print("Shape:", df.shape)
    print()
    print("AQI resumen:")
    print(df["aqi"].describe())


if __name__ == "__main__":
    main()