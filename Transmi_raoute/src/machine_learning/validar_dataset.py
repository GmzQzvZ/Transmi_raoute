import pandas as pd
from pathlib import Path


# ============================================================
# CONFIGURACIÓN
# ============================================================

# El proyecto TransMi Route está dos niveles por encima
# de la carpeta machine_learning
ROOT_DIR = Path(__file__).resolve().parents[2]

ARCHIVO = (
    ROOT_DIR
    / "data"
    / "processed"
    / "dataset_transmilenio_ml.csv"
)


# ============================================================
# VERIFICAR ARCHIVO
# ============================================================

print("=" * 60)
print(" VALIDACIÓN DATASET TRANSMI ROUTE")
print("=" * 60)

print(f"\nArchivo buscado:")
print(ARCHIVO)

if not ARCHIVO.exists():
    print("\nERROR: No se encontró el archivo.")
    print("Verifica que exista:")
    print(ARCHIVO)
    raise SystemExit(1)


# ============================================================
# CARGAR DATASET
# ============================================================

df = pd.read_csv(ARCHIVO)

print("\nDataset cargado correctamente.")


# ============================================================
# INFORMACIÓN GENERAL
# ============================================================

print(f"\nTotal registros: {len(df)}")

print("\nColumnas:")
for columna in df.columns:
    print(f"  - {columna}")


# ============================================================
# DISTRIBUCIÓN DE CLASES
# ============================================================

print("\nDistribución de clases:")
print(df["clasificacion"].value_counts())


# ============================================================
# DUPLICADOS EXACTOS
# ============================================================

duplicados = df.duplicated().sum()

print(f"\nDuplicados exactos: {duplicados}")


# ============================================================
# DUPLICADOS POR CARACTERÍSTICAS
# ============================================================

columnas_ruta = [
    "origen",
    "destino",
    "cantidad_estaciones",
    "cantidad_tramos",
    "cantidad_servicios",
    "cantidad_transbordos",
]

duplicados_caracteristicas = (
    df.duplicated(subset=columnas_ruta).sum()
)

print(
    "\nDuplicados por origen/destino/características: "
    f"{duplicados_caracteristicas}"
)


# ============================================================
# CONFLICTOS DE CLASIFICACIÓN
# ============================================================

grupo = (
    df.groupby(
        [
            "origen",
            "destino",
            "cantidad_estaciones",
            "cantidad_servicios",
            "cantidad_transbordos",
        ]
    )["clasificacion"]
    .nunique()
)

conflictos = grupo[grupo > 1]

print(
    "\nCasos donde las mismas características "
    "tienen clases diferentes:"
)

print(len(conflictos))


# ============================================================
# ESTADÍSTICAS
# ============================================================

print("\nEstadísticas:")

print(
    df[
        [
            "cantidad_estaciones",
            "cantidad_tramos",
            "cantidad_servicios",
            "cantidad_transbordos",
        ]
    ].describe()
)


# ============================================================
# RESULTADO
# ============================================================

print("\n" + "=" * 60)
print(" VALIDACIÓN TERMINADA")
print("=" * 60)

if conflictos.empty:
    print("\nOK: No existen conflictos de clasificación.")
else:
    print(
        f"\nATENCIÓN: Se encontraron "
        f"{len(conflictos)} conflictos."
    )

print("\nDataset listo para continuar con el entrenamiento.")