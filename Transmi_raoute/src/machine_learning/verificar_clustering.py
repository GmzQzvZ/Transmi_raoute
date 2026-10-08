
import json
from pathlib import Path

import numpy as np
import pandas as pd


# Raíz del proyecto: Transmi_raoute/
ROOT = Path(__file__).resolve().parents[2]

JSON_PATH = ROOT / "data" / "processed" / "rutas_orientadas.json"
CSV_RUTAS = ROOT / "data" / "ml" / "resultados" / "rutas_clusterizadas.csv"
CSV_RESUMEN = ROOT / "data" / "ml" / "resultados" / "resumen_clusters.csv"

TOLERANCIA = 0.02
REGISTROS_ESPERADOS = 99


def comprobar(condicion, mensaje):
    """Imprime el resultado de una prueba sin detener las demás."""
    estado = "OK" if condicion else "REVISAR"
    print(f"[{estado}] {mensaje}")
    return condicion


def main():
    print("\n=== PRUEBAS DE TRANS MI ROUTE: K-MEANS ===\n")

    # 1. Comprobar que los archivos existan.
    if not comprobar(JSON_PATH.exists(), f"Existe {JSON_PATH.name}"):
        print("No se puede continuar: falta el archivo JSON.")
        return

    if not comprobar(CSV_RUTAS.exists(), f"Existe {CSV_RUTAS.name}"):
        print("No se puede continuar: falta {CSV_RUTAS.name}.")
        return

    if not comprobar(CSV_RESUMEN.exists(), f"Existe {CSV_RESUMEN.name}"):
        print("No se puede continuar: falta {CSV_RESUMEN.name}.")
        return

    # 2. Leer archivos.
    rutas = pd.read_csv(CSV_RUTAS)
    resumen = pd.read_csv(CSV_RESUMEN)

    columnas_necesarias = {
        "cantidad_estaciones",
        "cantidad_tramos",
        "distancia_euclidiana",
        "cluster",
    }

    comprobar(
        columnas_necesarias.issubset(rutas.columns),
        "El CSV de rutas contiene las columnas necesarias."
    )

    if not columnas_necesarias.issubset(rutas.columns):
        print("Columnas encontradas:", list(rutas.columns))
        return

    # 3. Verificar cantidad de registros.
    comprobar(
        len(rutas) == REGISTROS_ESPERADOS,
        f"Hay {len(rutas)} registros; se esperaban "
        f"{REGISTROS_ESPERADOS}."
    )

    # 4. Verificar valores ausentes y valores no numéricos.
    columnas_numericas = [
        "cantidad_estaciones",
        "cantidad_tramos",
        "distancia_euclidiana",
        "cluster",
    ]

    for columna in columnas_numericas:
        valores = pd.to_numeric(rutas[columna], errors="coerce")
        comprobar(
            valores.notna().all(),
            f"{columna}: todos los valores son numéricos."
        )

        comprobar(
            np.isfinite(valores.dropna()).all(),
            f"{columna}: no hay valores infinitos."
        )

    # 5. Verificar que las medidas no sean negativas.
    for columna in [
        "cantidad_estaciones",
        "cantidad_tramos",
        "distancia_euclidiana",
    ]:
        valores = pd.to_numeric(rutas[columna], errors="coerce")
        comprobar(
            (valores.dropna() >= 0).all(),
            f"{columna}: no hay valores negativos."
        )

    # 6. Comprobar cuántos clústeres existen.
    print("\n--- Distribución de clústeres ---")
    print(rutas["cluster"].value_counts().sort_index())

    comprobar(
        rutas["cluster"].nunique() == 3,
        f"Se encontraron {rutas['cluster'].nunique()} clústeres."
    )

    # 7. Recalcular el resumen directamente desde las rutas.
    resumen_calculado = (
        rutas.groupby("cluster")
        .agg(
            cantidad_rutas=("cluster", "size"),
            cantidad_estaciones=("cantidad_estaciones", "mean"),
            cantidad_tramos=("cantidad_tramos", "mean"),
            distancia_euclidiana=("distancia_euclidiana", "mean"),
        )
        .reset_index()
    )

    print("\n--- Resumen recalculado desde las rutas ---")
    print(resumen_calculado.round(2).to_string(index=False))

    # 8. Comparar el resumen calculado con el CSV original.
    columnas_resumen = {
        "cluster",
        "cantidad_rutas",
        "cantidad_estaciones",
        "cantidad_tramos",
        "distancia_euclidiana",
    }

    if comprobar(
        columnas_resumen.issubset(resumen.columns),
        "El resumen tiene las columnas esperadas."
    ):
        comparacion = resumen_calculado.merge(
            resumen,
            on="cluster",
            how="outer",
            suffixes=("_calculado", "_archivo"),
            indicator=True,
        )

        comprobar(
            (comparacion["_merge"] == "both").all(),
            "Ambos archivos contienen los mismos identificadores de clúster."
        )

        for columna in [
            "cantidad_rutas",
            "cantidad_estaciones",
            "cantidad_tramos",
            "distancia_euclidiana",
        ]:
            a = comparacion[f"{columna}_calculado"]
            b = comparacion[f"{columna}_archivo"]

            comprobar(
                np.allclose(
                    a, b, atol=TOLERANCIA, rtol=0, equal_nan=True
                ),
                f"El resumen coincide para {columna}."
            )

    # 9. Verificar si cantidad_servicios aporta variación.
    if "cantidad_servicios" in rutas.columns:
        print("\n--- Variable cantidad_servicios ---")
        print(rutas["cantidad_servicios"].value_counts(dropna=False))

        comprobar(
            rutas["cantidad_servicios"].nunique(dropna=False) > 1,
            "cantidad_servicios presenta variación entre registros."
        )

        if rutas["cantidad_servicios"].nunique(dropna=False) == 1:
            print(
                "AVISO: la variable es constante y no distingue "
                "los registros. No aporta información para separarlos."
            )

    # 10. Revisar las coordenadas originales.
    print("\n--- Diagnóstico de coordenadas ---")

    with open(JSON_PATH, "r", encoding="utf-8") as archivo:
        datos = json.load(archivo)

    lista_rutas = datos.get("rutas", {})
    if isinstance(lista_rutas, dict):
        lista_rutas = lista_rutas.values()

    xs, ys = [], []
    estaciones_con_coordenadas = 0

    for ruta in lista_rutas:
        for estacion in ruta.get("estaciones_base", []):
            x = estacion.get("x")
            y = estacion.get("y")

            try:
                x = float(x)
                y = float(y)

                if np.isfinite(x) and np.isfinite(y):
                    xs.append(x)
                    ys.append(y)
                    estaciones_con_coordenadas += 1
            except (TypeError, ValueError):
                continue

    comprobar(
        estaciones_con_coordenadas > 0,
        f"Se encontraron {estaciones_con_coordenadas} estaciones "
        "con coordenadas numéricas."
    )

    if xs and ys:
        print(f"X: mínimo={min(xs):.6f}, máximo={max(xs):.6f}")
        print(f"Y: mínimo={min(ys):.6f}, máximo={max(ys):.6f}")

        parecen_geograficas = (
            all(-180 <= x <= 180 for x in xs)
            and all(-90 <= y <= 90 for y in ys)
        )

        if parecen_geograficas:
            print(
                "AVISO: los rangos son compatibles con coordenadas "
                "geográficas longitud/latitud, pero no lo demuestran."
            )
        else:
            print(
                "AVISO: los valores no parecen estar todos en los "
                "rangos habituales de longitud/latitud."
            )

        print(
            "La unidad de distancia no se puede confirmar solo "
            "con los valores: revisa la fuente y el sistema de "
            "coordenadas utilizado."
        )

    print("\n=== FIN DE LAS PRUEBAS ===")


if __name__ == "__main__":
    main()