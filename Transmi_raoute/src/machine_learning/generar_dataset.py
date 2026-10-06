"""
Generación del dataset de Machine Learning para TransMi Route.

El dataset se construye a partir de la red de estaciones y rutas
previamente procesadas.

No se inventan datos de tiempo, distancia, tráfico o demanda.
Las características se obtienen de los recorridos existentes.
"""

from __future__ import annotations

import csv
import json
import random
from collections import defaultdict, deque
from pathlib import Path


# ============================================================
# RUTAS DEL PROYECTO
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[2]

ARCHIVO_RUTAS = ROOT_DIR / "data" / "processed" / "rutas_orientadas.json"
ARCHIVO_SALIDA = ROOT_DIR / "data" / "processed" / "dataset_transmilenio_ml.csv"


# ============================================================
# CONFIGURACIÓN
# ============================================================

SEMILLA = 42

COLUMNAS = [
    "id_recorrido",
    "origen",
    "destino",
    "cantidad_estaciones",
    "cantidad_tramos",
    "cantidad_servicios",
    "cantidad_transbordos",
    "clasificacion",
]


# ============================================================
# CARGA DE DATOS
# ============================================================

def cargar_rutas():
    """Carga las rutas orientadas procesadas."""

    if not ARCHIVO_RUTAS.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo:\n{ARCHIVO_RUTAS}"
        )

    with open(ARCHIVO_RUTAS, "r", encoding="utf-8") as archivo:
        datos = json.load(archivo)

    return datos


# ============================================================
# NORMALIZACIÓN
# ============================================================

def extraer_registros(datos):
    """
    Intenta encontrar la lista de registros dentro del JSON.

    Permite trabajar tanto con una lista directamente como
    con estructuras que contienen una lista en alguna de sus claves.
    """

    if isinstance(datos, list):
        return datos

    if isinstance(datos, dict):

        posibles_claves = [
            "rutas",
            "direcciones",
            "data",
            "features",
            "resultados",
        ]

        for clave in posibles_claves:
            valor = datos.get(clave)

            if isinstance(valor, list):
                return valor

        # Si el diccionario ya representa un registro único.
        return [datos]

    raise ValueError("Formato de JSON no reconocido.")


def obtener_valor(registro, nombres, default=None):
    """Obtiene un valor buscando diferentes nombres posibles."""

    if not isinstance(registro, dict):
        return default

    for nombre in nombres:

        if nombre in registro:
            return registro[nombre]

        # Algunos datos pueden venir dentro de attributes.
        attributes = registro.get("attributes")

        if isinstance(attributes, dict) and nombre in attributes:
            return attributes[nombre]

    return default


# ============================================================
# CONSTRUCCIÓN DEL GRAFO
# ============================================================

def construir_grafo(registros):
    """
    Construye un grafo dirigido.

    Cada estación representa un nodo.
    Una conexión consecutiva entre estaciones representa una arista.

    El grafo almacena también el servicio asociado a cada conexión.
    """

    grafo = defaultdict(list)

    for registro in registros:

        estaciones = obtener_valor(
            registro,
            [
                "estaciones",
                "estaciones_orientadas",
                "paradas",
                "stops",
            ],
        )

        servicio = obtener_valor(
            registro,
            [
                "servicio",
                "nombre_ruta",
                "ruta",
                "servicio_unico",
            ],
            "DESCONOCIDO",
        )

        if not isinstance(estaciones, list):
            continue

        if len(estaciones) < 2:
            continue

        for i in range(len(estaciones) - 1):

            origen = str(estaciones[i])
            destino = str(estaciones[i + 1])

            grafo[origen].append(
                {
                    "destino": destino,
                    "servicio": str(servicio),
                }
            )

    return grafo


# ============================================================
# BFS
# ============================================================

def buscar_ruta_bfs(grafo, origen, destino):
    """
    Busca una ruta mediante BFS.

    Devuelve una lista de estaciones.
    """

    if origen not in grafo:
        return None

    cola = deque([[origen]])
    visitados = {origen}

    while cola:

        camino = cola.popleft()
        actual = camino[-1]

        if actual == destino:
            return camino

        for conexion in grafo.get(actual, []):

            siguiente = conexion["destino"]

            if siguiente in visitados:
                continue

            visitados.add(siguiente)

            nuevo_camino = camino + [siguiente]
            cola.append(nuevo_camino)

    return None


# ============================================================
# CARACTERÍSTICAS
# ============================================================

def contar_servicios(grafo, camino):
    """Cuenta los servicios diferentes utilizados."""

    servicios = set()

    for origen, destino in zip(camino, camino[1:]):

        conexiones = grafo.get(origen, [])

        for conexion in conexiones:

            if conexion["destino"] == destino:
                servicios.add(conexion["servicio"])
                break

    return len(servicios)


def calcular_caracteristicas(grafo, camino):
    """
    Calcula las características utilizadas por el modelo.
    """

    cantidad_estaciones = len(camino)

    cantidad_tramos = max(0, len(camino) - 1)

    cantidad_servicios = contar_servicios(
        grafo,
        camino,
    )

    cantidad_transbordos = max(
        0,
        cantidad_servicios - 1,
    )

    return {
        "cantidad_estaciones": cantidad_estaciones,
        "cantidad_tramos": cantidad_tramos,
        "cantidad_servicios": cantidad_servicios,
        "cantidad_transbordos": cantidad_transbordos,
    }


# ============================================================
# GENERACIÓN DEL DATASET
# ============================================================

def generar_dataset():

    random.seed(SEMILLA)

    print("==========================================")
    print(" GENERADOR DATASET TRANSMI ROUTE")
    print("==========================================")

    print(f"\nLeyendo: {ARCHIVO_RUTAS}")

    datos = cargar_rutas()
    registros = extraer_registros(datos)

    print(f"Registros encontrados: {len(registros)}")

    grafo = construir_grafo(registros)

    print(f"Estaciones/nodos encontrados: {len(grafo)}")

    estaciones = list(grafo.keys())

    if len(estaciones) < 2:
        raise RuntimeError(
            "No hay suficientes estaciones para generar recorridos."
        )

    filas = []

    contador = 1

    # --------------------------------------------------------
    # Generamos diferentes pares origen-destino
    # --------------------------------------------------------

    pares = []

    intentos = 0
    max_intentos = 5000

    while len(pares) < 500 and intentos < max_intentos:

        intentos += 1

        origen = random.choice(estaciones)
        destino = random.choice(estaciones)

        if origen == destino:
            continue

        par = (origen, destino)

        if par in pares:
            continue

        ruta = buscar_ruta_bfs(
            grafo,
            origen,
            destino,
        )

        if ruta is None:
            continue

        pares.append(par)

    print(f"Pares origen-destino válidos: {len(pares)}")

    # --------------------------------------------------------
    # Construimos las filas
    # --------------------------------------------------------

    for origen, destino in pares:

        ruta = buscar_ruta_bfs(
            grafo,
            origen,
            destino,
        )

        if not ruta:
            continue

        caracteristicas = calcular_caracteristicas(
            grafo,
            ruta,
        )

        fila = {
            "id_recorrido": contador,
            "origen": origen,
            "destino": destino,
            "cantidad_estaciones": caracteristicas[
                "cantidad_estaciones"
            ],
            "cantidad_tramos": caracteristicas[
                "cantidad_tramos"
            ],
            "cantidad_servicios": caracteristicas[
                "cantidad_servicios"
            ],
            "cantidad_transbordos": caracteristicas[
                "cantidad_transbordos"
            ],

            # En esta primera versión todos son recorridos
            # válidos generados por BFS.
            "clasificacion": "RECOMENDADA",
        }

        filas.append(fila)
        contador += 1

    # --------------------------------------------------------
    # Guardar CSV
    # --------------------------------------------------------

    ARCHIVO_SALIDA.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        ARCHIVO_SALIDA,
        "w",
        newline="",
        encoding="utf-8",
    ) as archivo:

        escritor = csv.DictWriter(
            archivo,
            fieldnames=COLUMNAS,
        )

        escritor.writeheader()
        escritor.writerows(filas)

    print("\n==========================================")
    print(" DATASET GENERADO")
    print("==========================================")

    print(f"Filas: {len(filas)}")
    print(f"Archivo: {ARCHIVO_SALIDA}")

    print("\nColumnas:")

    for columna in COLUMNAS:
        print(f"  - {columna}")

    return filas


# ============================================================
# EJECUCIÓN
# ============================================================

if __name__ == "__main__":
    generar_dataset()