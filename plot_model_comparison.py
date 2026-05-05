import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def main():
    input_path = Path("reports/model_comparison.csv")
    output_dir = Path("reports/figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)

    print("Resultados cargados:")
    print(df)

    # Gráfico 1: Accuracy por modelo
    plt.figure(figsize=(10, 5))
    plt.bar(df["model"], df["accuracy"])
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Accuracy")
    plt.title("Comparación de accuracy por modelo")
    plt.tight_layout()
    plt.savefig(output_dir / "accuracy_comparison.png", dpi=300)
    plt.close()

    # Gráfico 2: Macro F1 por modelo
    plt.figure(figsize=(10, 5))
    plt.bar(df["model"], df["macro_f1"])
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Macro F1-score")
    plt.title("Comparación de Macro F1-score por modelo")
    plt.tight_layout()
    plt.savefig(output_dir / "macro_f1_comparison.png", dpi=300)
    plt.close()

    # Gráfico 3: Recall de clase buena
    plt.figure(figsize=(10, 5))
    plt.bar(df["model"], df["recall_buena"])
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Recall clase buena")
    plt.title("Comparación de recall para clase minoritaria 'buena'")
    plt.tight_layout()
    plt.savefig(output_dir / "recall_buena_comparison.png", dpi=300)
    plt.close()

    print()
    print("Gráficos guardados en:")
    print(output_dir / "accuracy_comparison.png")
    print(output_dir / "macro_f1_comparison.png")
    print(output_dir / "recall_buena_comparison.png")


if __name__ == "__main__":
    main()