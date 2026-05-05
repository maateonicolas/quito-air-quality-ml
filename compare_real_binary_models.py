import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def main():
    output_dir = Path("reports/figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    results = [
        {
            "model": "Logistic Regression",
            "accuracy": 0.7066,
            "f1_buena": 0.75,
            "f1_no_buena": 0.65,
            "macro_f1": 0.70,
        },
        {
            "model": "Decision Tree",
            "accuracy": 0.6653,
            "f1_buena": 0.67,
            "f1_no_buena": 0.66,
            "macro_f1": 0.67,
        },
        {
            "model": "Random Forest",
            "accuracy": 0.7603,
            "f1_buena": 0.80,
            "f1_no_buena": 0.70,
            "macro_f1": 0.75,
        },
        {
            "model": "Neural Network",
            "accuracy": 0.7355,
            "f1_buena": 0.78,
            "f1_no_buena": 0.67,
            "macro_f1": 0.72,
        },
    ]

    df = pd.DataFrame(results)
    df.to_csv("reports/real_binary_model_comparison.csv", index=False)

    print(df)

    plt.figure(figsize=(8, 5))
    plt.bar(df["model"], df["accuracy"])
    plt.ylabel("Accuracy")
    plt.title("Accuracy en datos reales - clasificación binaria")
    plt.tight_layout()
    plt.savefig(output_dir / "real_binary_accuracy.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.bar(df["model"], df["macro_f1"])
    plt.ylabel("Macro F1-score")
    plt.title("Macro F1-score en datos reales - clasificación binaria")
    plt.tight_layout()
    plt.savefig(output_dir / "real_binary_macro_f1.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8, 5))
    x = range(len(df))
    plt.bar([i - 0.2 for i in x], df["f1_buena"], width=0.4, label="F1 buena")
    plt.bar([i + 0.2 for i in x], df["f1_no_buena"], width=0.4, label="F1 no_buena")
    plt.xticks(x, df["model"])
    plt.ylabel("F1-score")
    plt.title("F1-score por clase en datos reales")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "real_binary_f1_by_class.png", dpi=300)
    plt.close()

    print()
    print("Archivos generados:")
    print("reports/real_binary_model_comparison.csv")
    print("reports/figures/real_binary_accuracy.png")
    print("reports/figures/real_binary_macro_f1.png")
    print("reports/figures/real_binary_f1_by_class.png")


if __name__ == "__main__":
    main()