"""
Entrenamiento del árbol de decisión para TransMi Route.
"""

from pathlib import Path

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)


# ============================================================
# RUTAS
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[2]

ARCHIVO_DATASET = (
    ROOT_DIR
    / "data"
    / "processed"
    / "dataset_transmilenio_ml.csv"
)


# ============================================================
# COLUMNAS
# ============================================================

CARACTERISTICAS = [
    "cantidad_estaciones",
    "cantidad_tramos",
    "cantidad_servicios",
    "cantidad_transbordos",
]

OBJETIVO = "clasificacion"


# ============================================================
# ENTRENAMIENTO
# ============================================================

def entrenar():

    if not ARCHIVO_DATASET.exists():

        raise FileNotFoundError(
            f"No se encontró el dataset:\n{ARCHIVO_DATASET}\n\n"
            "Ejecuta primero generar_dataset.py"
        )

    print("==========================================")
    print(" ENTRENAMIENTO ÁRBOL DE DECISIÓN")
    print("==========================================")

    df = pd.read_csv(
        ARCHIVO_DATASET
    )

    print(f"\nRegistros: {len(df)}")

    print("\nClases encontradas:")

    print(
        df[OBJETIVO]
        .value_counts()
    )

    # --------------------------------------------------------
    # Validación básica
    # --------------------------------------------------------

    clases = df[OBJETIVO].nunique()

    if clases < 2:

        print(
            "\n⚠️ El dataset todavía tiene una sola clase."
        )

        print(
            "Necesitamos generar rutas "
            "RECOMENDADAS y NO_RECOMENDADAS "
            "antes de entrenar el modelo."
        )

        return None

    # --------------------------------------------------------
    # Datos de entrada
    # --------------------------------------------------------

    X = df[CARACTERISTICAS]

    y = df[OBJETIVO]

    # --------------------------------------------------------
    # Separación entrenamiento/prueba
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    print(f"\nDatos entrenamiento: {len(X_train)}")
    print(f"Datos prueba: {len(X_test)}")

    # --------------------------------------------------------
    # Modelo
    # --------------------------------------------------------

    modelo = DecisionTreeClassifier(
        max_depth=4,
        random_state=42,
    )

    modelo.fit(
        X_train,
        y_train,
    )

    # --------------------------------------------------------
    # Predicción
    # --------------------------------------------------------

    predicciones = modelo.predict(
        X_test
    )

    # --------------------------------------------------------
    # Resultados
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        predicciones,
    )

    print("\n==========================================")
    print(" RESULTADOS")
    print("==========================================")

    print(
        f"\nAccuracy: {accuracy:.4f}"
    )

    print(
        "\nReporte de clasificación:"
    )

    print(
        classification_report(
            y_test,
            predicciones,
            zero_division=0,
        )
    )

    print(
        "Matriz de confusión:"
    )

    print(
        confusion_matrix(
            y_test,
            predicciones,
        )
    )

    return modelo


# ============================================================
# EJECUCIÓN
# ============================================================

if __name__ == "__main__":
    entrenar()