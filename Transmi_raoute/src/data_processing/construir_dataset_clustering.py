import csv
import json
import math
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[2]

PROCESSED_DIR = BASE_DIR / "data" / "processed"
ML_DIR = BASE_DIR / "data" / "ml"

ARCHIVO_RUTAS = PROCESSED_DIR / "rutas_orientadas.json"
ARCHIVO_SALIDA = ML_DIR / "dataset_clustering.csv"


def cargar_json(ruta):
    if not ruta.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo:\n{ruta}"
        )

    with ruta.open("r", encoding="utf-8") as archivo:
        return json.load(archivo)


def calcular_distancia(estaciones):
    """
    Calcula una distancia aproximada acumulando
    la distancia euclidiana entre estaciones consecutivas.

    Utiliza las coordenadas X/Y disponibles en los datos.
    """

    distancia_total = 0.0

    for anterior, actual in zip(
        estaciones,
        estaciones[1:]
    ):

        try:
            x1 = float(anterior["x"])
            y1 = float(anterior["y"])
            x2 = float(actual["x"])
            y2 = float(actual["y"])
        except (KeyError, TypeError, ValueError):
            continue

        distancia = math.sqrt(
            (x2 - x1) ** 2 +
            (y2 - y1) ** 2
        )

        distancia_total += distancia

    return round(distancia_total, 2)


def construir_registros(datos):
    """
    Convierte rutas_orientadas.json en un dataset
    preparado para aprendizaje no supervisado.

    Cada registro representa una dirección construida
    de una ruta de TransMilenio.
    """

    registros = []

    rutas = datos.get("rutas", {})

    for id_ruta, ruta in rutas.items():

        nombre_ruta = ruta.get(
            "nombre_ruta",
            id_ruta
        )

        servicio = ruta.get(
            "servicio",
            ""
        )

        direcciones = ruta.get(
            "direcciones",
            []
        )

        for numero, direccion in enumerate(
            direcciones,
            start=1
        ):

            if direccion.get("estado") != "construida":
                continue

            estaciones = direccion.get(
                "estaciones_orientadas",
                []
            )

            if not isinstance(estaciones, list):
                continue

            cantidad_estaciones = len(
                estaciones
            )

            cantidad_tramos = max(
                cantidad_estaciones - 1,
                0
            )

            distancia = calcular_distancia(
                estaciones
            )

            registros.append({
                "id_ruta": str(id_ruta),
                "nombre_ruta": str(nombre_ruta),
                "servicio": str(servicio),
                "direccion": numero,
                "cantidad_estaciones":
                    cantidad_estaciones,
                "cantidad_tramos":
                    cantidad_tramos,
                "distancia_euclidiana":
                    distancia,
                "cantidad_servicios": 1,
            })

    return registros


def guardar_csv(registros):

    ML_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    columnas = [
        "id_ruta",
        "nombre_ruta",
        "servicio",
        "direccion",
        "cantidad_estaciones",
        "cantidad_tramos",
        "distancia_euclidiana",
        "cantidad_servicios",
    ]

    with ARCHIVO_SALIDA.open(
        "w",
        encoding="utf-8",
        newline=""
    ) as archivo:

        escritor = csv.DictWriter(
            archivo,
            fieldnames=columnas
        )

        escritor.writeheader()
        escritor.writerows(registros)


def mostrar_resumen(registros):

    print()
    print("=" * 70)
    print("🚍 TRANSMI ROUTE - DATASET PARA K-MEANS")
    print("=" * 70)

    print()
    print("📊 REGISTROS")
    print("-" * 70)

    print(
        f"Registros utilizados: {len(registros)}"
    )

    if not registros:
        print(
            "⚠️ No se encontraron registros."
        )
        return

    total_estaciones = sum(
        registro["cantidad_estaciones"]
        for registro in registros
    )

    total_tramos = sum(
        registro["cantidad_tramos"]
        for registro in registros
    )

    total_distancia = sum(
        registro["distancia_euclidiana"]
        for registro in registros
    )

    servicios = sorted(
        {
            registro["servicio"]
            for registro in registros
        }
    )

    print(
        f"Total de estaciones: "
        f"{total_estaciones}"
    )

    print(
        f"Total de tramos: "
        f"{total_tramos}"
    )

    print(
        f"Distancia acumulada X/Y: "
        f"{total_distancia:.2f}"
    )

    print(
        f"Servicios encontrados: "
        f"{len(servicios)}"
    )

    print(
        f"Valores de servicio: "
        f"{', '.join(servicios)}"
    )

    print()
    print("🔢 VARIABLES NUMÉRICAS PARA K-MEANS")
    print("-" * 70)

    print("1. cantidad_estaciones")
    print("2. cantidad_tramos")
    print("3. distancia_euclidiana")

    print()
    print("📝 VARIABLES DESCRIPTIVAS")
    print("-" * 70)

    print("• id_ruta")
    print("• nombre_ruta")
    print("• servicio")
    print("• direccion")

    print()
    print("💾 ARCHIVO GENERADO")
    print("-" * 70)

    print(ARCHIVO_SALIDA)

    print()
    print("=" * 70)
    print("✅ Dataset construido correctamente")
    print("=" * 70)
    print()


def main():

    print()
    print("🔎 Cargando datos procesados...")

    datos = cargar_json(
        ARCHIVO_RUTAS
    )

    registros = construir_registros(
        datos
    )

    guardar_csv(
        registros
    )

    mostrar_resumen(
        registros
    )


if __name__ == "__main__":
    main()