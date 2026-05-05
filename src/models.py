from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


def build_logistic_model(preprocessor):
    """
    Modelo baseline: Regresión Logística.
    class_weight='balanced' ayuda con clases desbalanceadas.
    """
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", LogisticRegression(
                max_iter=1000,
                class_weight="balanced"
            ))
        ]
    )


def build_tree_model(preprocessor):
    """
    Árbol de decisión.
    """
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", DecisionTreeClassifier(
                max_depth=4,
                random_state=42,
                class_weight="balanced"
            ))
        ]
    )


def build_random_forest_model(preprocessor):
    """
    Random Forest.
    """
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", RandomForestClassifier(
                n_estimators=200,
                max_depth=8,
                random_state=42,
                class_weight="balanced"
            ))
        ]
    )