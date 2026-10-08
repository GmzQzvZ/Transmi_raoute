"""Agrupa las rutas del dataset y guarda los resultados del análisis."""

from pathlib import Path

import matplotlib

# Permite guardar la gráfica incluso cuando el script se ejecuta sin interfaz.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


ROOT_DIR = Path(__file__).resolve().parents[2]
ARCHIVO_ENTRADA = ROOT_DIR / "data" / "ml" / "dataset_clustering.csv"
DIR_RESULTADOS = ROOT_DIR / "data" / "ml" / "resultados"
ARCHIVO_RUTAS = DIR_RESULTADOS / "rutas_clusterizadas.csv"
ARCHIVO_RESUMEN = DIR_RESULTADOS / "resumen_clusters.csv"
ARCHIVO_GRAFICA = DIR_RESULTADOS / "metodo_codo.png"

# Estas columnas describen las características de una ruta. Los identificadores
# y nombres no se usan para agrupar, porque no representan magnitudes.
CARACTERISTICAS = [
    "cantidad_estaciones",
    "cantidad_tramos",
    "distancia_euclidiana",
    "cantidad_servicios",
]


def ejecutar_clustering() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Lee el dataset, asigna clusters y guarda CSV y gráfica del codo."""
    if not ARCHIVO_ENTRADA.exists():
        raise FileNotFoundError(
            f"No se encontró el dataset de entrada: {ARCHIVO_ENTRADA}"
        )

    datos = pd.read_csv(ARCHIVO_ENTRADA)
    faltantes = [columna for columna in CARACTERISTICAS if columna not in datos.columns]
    if faltantes:
        raise ValueError(
            "Al dataset le faltan columnas requeridas: " + ", ".join(faltantes)
        )
    if datos.empty:
        raise ValueError(f"El dataset no contiene filas: {ARCHIVO_ENTRADA}")

    datos[CARACTERISTICAS] = datos[CARACTERISTICAS].apply(
        pd.to_numeric, errors="coerce"
    )
    datos = datos.dropna(subset=CARACTERISTICAS).copy()
    if datos.empty:
        raise ValueError("No hay filas con valores numéricos completos para agrupar.")

    DIR_RESULTADOS.mkdir(parents=True, exist_ok=True)
    valores = datos[CARACTERISTICAS]
    escalados = StandardScaler().fit_transform(valores)

    # Calcula inercias para k=1..10 y usa k=3 para las etiquetas finales,
    # limitando k al número de registros disponibles.
    max_k = min(10, len(datos))
    inercias = []
    for k in range(1, max_k + 1):
        modelo = KMeans(n_clusters=k, random_state=42, n_init=10)
        modelo.fit(escalados)
        inercias.append(modelo.inertia_)

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, max_k + 1), inercias, marker="o")
    plt.title("Método del codo para clustering de rutas")
    plt.xlabel("Número de clusters (k)")
    plt.ylabel("Inercia")
    plt.xticks(range(1, max_k + 1))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(ARCHIVO_GRAFICA, dpi=150)
    plt.close()

    k_final = min(3, len(datos))
    modelo_final = KMeans(n_clusters=k_final, random_state=42, n_init=10)
    datos["cluster"] = modelo_final.fit_predict(escalados)
    datos.to_csv(ARCHIVO_RUTAS, index=False, encoding="utf-8-sig")

    resumen = (
        datos.groupby("cluster")
        .agg(
            cantidad_rutas=("cluster", "size"),
            **{columna: (columna, "mean") for columna in CARACTERISTICAS},
        )
        .reset_index()
    )
    resumen.to_csv(ARCHIVO_RESUMEN, index=False, encoding="utf-8-sig")

    print(f"Rutas agrupadas: {ARCHIVO_RUTAS}")
    print(f"Resumen de clusters: {ARCHIVO_RESUMEN}")
    print(f"Gráfica del método del codo: {ARCHIVO_GRAFICA}")
    print(f"Registros procesados: {len(datos)} | Clusters asignados: {k_final}")
    return datos, resumen


if __name__ == "__main__":
    ejecutar_clustering()
