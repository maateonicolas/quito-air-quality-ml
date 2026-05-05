import pandas as pd

results = [
    {
        "model": "Logistic Regression",
        "accuracy": 0.8833,
        "recall_buena": 0.21,
        "recall_mala": 0.87,
        "recall_moderada": 0.94,
        "macro_f1": 0.71,
        "comment": "Mejor accuracy, pero bajo recall para clase buena."
    },
    {
        "model": "Decision Tree",
        "accuracy": 0.8433,
        "recall_buena": 0.14,
        "recall_mala": 0.88,
        "recall_moderada": 0.87,
        "macro_f1": 0.65,
        "comment": "Modelo simple, menor desempeño general."
    },
    {
        "model": "Random Forest",
        "accuracy": 0.8567,
        "recall_buena": 0.57,
        "recall_mala": 0.87,
        "recall_moderada": 0.87,
        "macro_f1": 0.77,
        "comment": "Mejor balance general entre clases."
    },
    {
        "model": "Neural Network",
        "accuracy": 0.8600,
        "recall_buena": 0.14,
        "recall_mala": 0.84,
        "recall_moderada": 0.92,
        "macro_f1": 0.66,
        "comment": "Buen accuracy, pero ignora clase minoritaria."
    },
    {
        "model": "Neural Network + Class Weights",
        "accuracy": 0.8367,
        "recall_buena": 0.50,
        "recall_mala": 0.86,
        "recall_moderada": 0.85,
        "macro_f1": 0.71,
        "comment": "Mejora la clase buena sacrificando accuracy."
    }
]

df = pd.DataFrame(results)

output_path = "reports/model_comparison.csv"
df.to_csv(output_path, index=False)

print(df)
print()
print(f"Resultados guardados en: {output_path}")