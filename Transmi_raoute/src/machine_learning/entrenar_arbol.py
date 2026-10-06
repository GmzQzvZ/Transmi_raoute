import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)


# ============================================================
# CONFIGURACIÓN
# ============================================================

# Ubicación del proyecto
ROOT_DIR = Path(__file__).resolve().parent.parent / "Transmi_raoute"

# Dataset
ARCHIVO_DATASET = (
    ROOT_DIR
    / "data"
    / "processed"
    / "dataset_transmilenio_ml.csv"
)

# Carpeta donde guardaremos los resultados
CARPETA_RESULTADOS = (
    ROOT_DIR
    / "data"
    / "processed"
    / "ml_resultados"
)

CARPETA_RESULTADOS.mkdir(parents=True, exist_ok=True)


# ============================================================
# CARGAR DATASET
# ============================================================

print("=" * 70)
print(" ENTRENAMIENTO DEL ÁRBOL DE DECISIÓN - TRANSMI ROUTE")
print("=" * 70)

print("\nCargando dataset...")

if not ARCHIVO_DATASET.exists():
    print("\nERROR: No se encontró el dataset:")
    print(ARCHIVO_DATASET)
    raise SystemExit(1)

df = pd.read_csv(ARCHIVO_DATASET)

print(f"Dataset cargado correctamente.")
print(f"Total de registros: {len(df)}")


# ============================================================
# CARACTERÍSTICAS Y VARIABLE OBJETIVO
# ============================================================

# Estas son las características que el árbol utilizará.
#
# cantidad_estaciones:
# Número de estaciones del recorrido.
#
# cantidad_servicios:
# Cantidad de servicios utilizados.
#
# cantidad_transbordos:
# Cantidad de transbordos realizados.

CARACTERISTICAS = [
    "cantidad_estaciones",
    "cantidad_servicios",
    "cantidad_transbordos",
]

OBJETIVO = "clasificacion"

X = df[CARACTERISTICAS]
y = df[OBJETIVO]


print("\nCaracterísticas utilizadas:")
for caracteristica in CARACTERISTICAS:
    print(f"  - {caracteristica}")

print(f"\nVariable objetivo: {OBJETIVO}")


# ============================================================
# DISTRIBUCIÓN DE CLASES
# ============================================================

print("\nDistribución de clases:")

print(y.value_counts())


# ============================================================
# DIVISIÓN DEL DATASET
# ============================================================

print("\nDividiendo dataset en entrenamiento y prueba...")

X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

print(f"Registros para entrenamiento: {len(X_entrenamiento)}")
print(f"Registros para prueba: {len(X_prueba)}")


# ============================================================
# CREAR MODELO
# ============================================================

print("\nCreando árbol de decisión...")

modelo = DecisionTreeClassifier(
    criterion="entropy",
    max_depth=4,
    random_state=42,
)


# ============================================================
# ENTRENAMIENTO
# ============================================================

print("Entrenando modelo...")

modelo.fit(
    X_entrenamiento,
    y_entrenamiento,
)

print("Modelo entrenado correctamente.")


# ============================================================
# PREDICCIONES
# ============================================================

print("\nGenerando predicciones...")

y_prediccion = modelo.predict(X_prueba)


# ============================================================
# EXACTITUD
# ============================================================

exactitud = accuracy_score(
    y_prueba,
    y_prediccion,
)

print("\n" + "=" * 70)
print(" RESULTADOS DEL MODELO")
print("=" * 70)

print(f"\nExactitud del modelo: {exactitud:.4f}")
print(f"Porcentaje de exactitud: {exactitud * 100:.2f}%")


# ============================================================
# REPORTE DE CLASIFICACIÓN
# ============================================================

print("\nReporte de clasificación:")

reporte = classification_report(
    y_prueba,
    y_prediccion,
)

print(reporte)


# ============================================================
# MATRIZ DE CONFUSIÓN
# ============================================================

matriz = confusion_matrix(
    y_prueba,
    y_prediccion,
    labels=[
        "RECOMENDADA",
        "NO_RECOMENDADA",
    ],
)

print("\nMatriz de confusión:")

print(matriz)


# ============================================================
# IMPORTANCIA DE LAS CARACTERÍSTICAS
# ============================================================

print("\nImportancia de las características:")

importancias = pd.DataFrame(
    {
        "caracteristica": CARACTERISTICAS,
        "importancia": modelo.feature_importances_,
    }
)

importancias = importancias.sort_values(
    by="importancia",
    ascending=False,
)

print(importancias.to_string(index=False))


# ============================================================
# GUARDAR IMPORTANCIA EN CSV
# ============================================================

archivo_importancias = (
    CARPETA_RESULTADOS
    / "importancia_caracteristicas.csv"
)

importancias.to_csv(
    archivo_importancias,
    index=False,
)

print(
    f"\nImportancia guardada en:"
    f"\n{archivo_importancias}"
)


# ============================================================
# GRÁFICA DE MATRIZ DE CONFUSIÓN
# ============================================================

print("\nGenerando matriz de confusión...")

disp = ConfusionMatrixDisplay(
    confusion_matrix=matriz,
    display_labels=[
        "RECOMENDADA",
        "NO_RECOMENDADA",
    ],
)

disp.plot()

plt.title("Matriz de confusión - TransMi Route")
plt.tight_layout()

archivo_matriz = (
    CARPETA_RESULTADOS
    / "matriz_confusion.png"
)

plt.savefig(
    archivo_matriz,
    dpi=150,
)

plt.show()

plt.close()


# ============================================================
# GRÁFICA DE IMPORTANCIA
# ============================================================

print("Generando gráfica de importancia...")

plt.figure(figsize=(8, 5))

plt.bar(
    importancias["caracteristica"],
    importancias["importancia"],
)

plt.title(
    "Importancia de características - TransMi Route"
)

plt.xlabel("Característica")
plt.ylabel("Importancia")

plt.xticks(
    rotation=20,
    ha="right",
)

plt.tight_layout()

archivo_importancia_grafica = (
    CARPETA_RESULTADOS
    / "importancia_caracteristicas.png"
)

plt.savefig(
    archivo_importancia_grafica,
    dpi=150,
)

plt.show()

plt.close()


# ============================================================
# GRÁFICA DEL ÁRBOL
# ============================================================

print("Generando visualización del árbol...")

plt.figure(figsize=(18, 10))

plot_tree(
    modelo,
    feature_names=CARACTERISTICAS,
    class_names=[
        "NO_RECOMENDADA",
        "RECOMENDADA",
    ],
    filled=True,
    rounded=True,
    fontsize=9,
)

plt.title(
    "Árbol de decisión - TransMi Route"
)

plt.tight_layout()

archivo_arbol = (
    CARPETA_RESULTADOS
    / "arbol_decision.png"
)

plt.savefig(
    archivo_arbol,
    dpi=150,
    bbox_inches="tight",
)

plt.show()

plt.close()


# ============================================================
# GUARDAR PREDICCIONES
# ============================================================

resultado_prueba = X_prueba.copy()

resultado_prueba["real"] = y_prueba
resultado_prueba["prediccion"] = y_prediccion

resultado_prueba = resultado_prueba.reset_index(
    drop=True
)

archivo_predicciones = (
    CARPETA_RESULTADOS
    / "predicciones_prueba.csv"
)

resultado_prueba.to_csv(
    archivo_predicciones,
    index=False,
)


# ============================================================
# GUARDAR MODELO
# ============================================================

import joblib

archivo_modelo = (
    CARPETA_RESULTADOS
    / "arbol_transmi_route.joblib"
)

joblib.dump(
    modelo,
    archivo_modelo,
)


# ============================================================
# RESUMEN FINAL
# ============================================================

print("\n" + "=" * 70)
print(" ENTRENAMIENTO TERMINADO")
print("=" * 70)

print(f"""
Registros totales:       {len(df)}
Registros entrenamiento: {len(X_entrenamiento)}
Registros prueba:        {len(X_prueba)}

Características:
- cantidad_estaciones
- cantidad_servicios
- cantidad_transbordos

Exactitud:
{exactitud * 100:.2f}%

Archivos generados:

1. Modelo:
{archivo_modelo}

2. Matriz de confusión:
{archivo_matriz}

3. Importancia de características:
{archivo_importancia_grafica}

4. Árbol de decisión:
{archivo_arbol}

5. Predicciones:
{archivo_predicciones}

6. Datos de importancia:
{archivo_importancias}
""")

print("=" * 70)
print(" PROCESO COMPLETADO CORRECTAMENTE")
print("=" * 70)