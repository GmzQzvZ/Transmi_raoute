import csv
import json
import random
from pathlib import Path
from collections import defaultdict, deque


# ============================================================
# CONFIGURACIÓN
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[1]

ARCHIVO_RUTAS = (
    ROOT_DIR
    / "data"
    / "processed"
    / "rutas_orientadas.json"
)

ARCHIVO_SALIDA = (
    ROOT_DIR
    / "data"
    / "processed"
    / "dataset_transmilenio_ml.csv"
)

MAX_PARES = 150
MAX_RUTAS_POR_PAR = 4

SEMILLA = 42


# ============================================================
# CARGAR RUTAS
# ============================================================

def cargar_datos():

    print(f"Leyendo: {ARCHIVO_RUTAS}")

    with open(
        ARCHIVO_RUTAS,
        "r",
        encoding="utf-8"
    ) as archivo:

        datos = json.load(archivo)

    rutas = datos.get("rutas", {})

    if not isinstance(rutas, dict):

        raise RuntimeError(
            "La estructura 'rutas' no es válida."
        )

    print(
        f"Rutas encontradas: {len(rutas)}"
    )

    return rutas


# ============================================================
# CONSTRUIR GRAFO
# ============================================================

def construir_grafo(rutas):

    grafo = defaultdict(list)

    estaciones = {}

    for id_ruta, ruta in rutas.items():

        servicio = str(
            ruta.get(
                "servicio",
                id_ruta
            )
        )

        for direccion in ruta.get(
            "direcciones",
            []
        ):

            if direccion.get(
                "estado"
            ) != "construida":

                continue

            estaciones_orientadas = (
                direccion.get(
                    "estaciones_orientadas",
                    []
                )
            )

            for estacion in estaciones_orientadas:

                estacion_id = estacion.get(
                    "id"
                )

                if estacion_id is None:
                    continue

                estaciones[estacion_id] = {
                    "id": estacion_id,
                    "nombre": estacion.get(
                        "nombre",
                        ""
                    )
                }

            for i in range(
                len(estaciones_orientadas) - 1
            ):

                actual = estaciones_orientadas[i]

                siguiente = (
                    estaciones_orientadas[i + 1]
                )

                origen = actual.get("id")
                destino = siguiente.get("id")

                if (
                    origen is None
                    or destino is None
                ):
                    continue

                grafo[origen].append(
                    {
                        "destino": destino,
                        "servicio": servicio,
                        "ruta": id_ruta,
                    }
                )

    return grafo, estaciones


# ============================================================
# LIMPIAR GRAFO
# ============================================================

def limpiar_grafo(grafo):

    limpio = defaultdict(list)

    for origen, conexiones in grafo.items():

        vistos = set()

        for conexion in conexiones:

            clave = (
                conexion["destino"],
                conexion["servicio"],
                conexion["ruta"],
            )

            if clave in vistos:
                continue

            vistos.add(clave)

            limpio[origen].append(
                conexion
            )

    return limpio


# ============================================================
# BFS CON BLOQUEOS
# ============================================================

def buscar_ruta_bfs(
    grafo,
    origen,
    destino,
    bloqueos=None
):

    if bloqueos is None:
        bloqueos = set()

    cola = deque()

    cola.append(
        (
            origen,
            [origen],
            []
        )
    )

    # La ruta importa más que visitar
    # simplemente el nodo.
    visitados = {
        origen
    }

    while cola:

        actual, camino, conexiones = (
            cola.popleft()
        )

        if actual == destino:

            return {
                "estaciones": camino,
                "conexiones": conexiones,
            }

        for conexion in grafo.get(
            actual,
            []
        ):

            siguiente = conexion[
                "destino"
            ]

            clave_bloqueo = (
                actual,
                siguiente,
                conexion["servicio"]
            )

            if clave_bloqueo in bloqueos:
                continue

            if siguiente in visitados:
                continue

            visitados.add(
                siguiente
            )

            cola.append(
                (
                    siguiente,
                    camino + [siguiente],
                    conexiones + [conexion],
                )
            )

    return None


# ============================================================
# IDENTIFICADOR DE RUTA
# ============================================================

def obtener_identificador_ruta(resultado):

    return tuple(
        conexion["destino"]
        for conexion
        in resultado["conexiones"]
    )


# ============================================================
# GENERAR RUTAS ALTERNATIVAS
# ============================================================

def generar_rutas_alternativas(
    grafo,
    origen,
    destino,
    max_rutas=4
):

    rutas_encontradas = []

    identificadores = set()

    # --------------------------------------------------------
    # Primera ruta
    # --------------------------------------------------------

    primera = buscar_ruta_bfs(
        grafo,
        origen,
        destino
    )

    if primera is None:
        return []

    rutas_encontradas.append(
        primera
    )

    identificadores.add(
        obtener_identificador_ruta(
            primera
        )
    )

    # --------------------------------------------------------
    # Buscar alternativas
    # --------------------------------------------------------

    bloqueos = set()

    while (
        len(rutas_encontradas)
        < max_rutas
    ):

        nuevas = []

        for ruta in list(
            rutas_encontradas
        ):

            for conexion in ruta[
                "conexiones"
            ]:

                bloqueo = (
                    ruta["estaciones"][
                        ruta["conexiones"].index(
                            conexion
                        )
                    ],
                    conexion["destino"],
                    conexion["servicio"]
                )

                if bloqueo in bloqueos:
                    continue

                bloqueos.add(
                    bloqueo
                )

                alternativa = (
                    buscar_ruta_bfs(
                        grafo,
                        origen,
                        destino,
                        bloqueos
                    )
                )

                if alternativa is None:
                    continue

                identificador = (
                    obtener_identificador_ruta(
                        alternativa
                    )
                )

                if identificador in identificadores:
                    continue

                identificadores.add(
                    identificador
                )

                nuevas.append(
                    alternativa
                )

                if (
                    len(rutas_encontradas)
                    + len(nuevas)
                    >= max_rutas
                ):
                    break

            if (
                len(rutas_encontradas)
                + len(nuevas)
                >= max_rutas
            ):
                break

        if not nuevas:
            break

        rutas_encontradas.extend(
            nuevas
        )

    return rutas_encontradas[
        :max_rutas
    ]


# ============================================================
# CARACTERÍSTICAS
# ============================================================

def calcular_caracteristicas(
    resultado
):

    estaciones = resultado[
        "estaciones"
    ]

    conexiones = resultado[
        "conexiones"
    ]

    servicios = set()

    for conexion in conexiones:

        servicios.add(
            conexion["servicio"]
        )

    cantidad_estaciones = len(
        estaciones
    )

    cantidad_tramos = len(
        conexiones
    )

    cantidad_servicios = len(
        servicios
    )

    cantidad_transbordos = max(
        0,
        cantidad_servicios - 1
    )

    return {
        "cantidad_estaciones":
            cantidad_estaciones,

        "cantidad_tramos":
            cantidad_tramos,

        "cantidad_servicios":
            cantidad_servicios,

        "cantidad_transbordos":
            cantidad_transbordos,
    }


# ============================================================
# EVALUAR RUTA
# ============================================================

def clave_evaluacion(
    caracteristicas
):

    return (
        caracteristicas[
            "cantidad_transbordos"
        ],

        caracteristicas[
            "cantidad_estaciones"
        ],

        caracteristicas[
            "cantidad_servicios"
        ],
    )


# ============================================================
# OBTENER ESTACIONES
# ============================================================

def obtener_estaciones(grafo):

    estaciones = set(
        grafo.keys()
    )

    for conexiones in grafo.values():

        for conexion in conexiones:

            estaciones.add(
                conexion["destino"]
            )

    return sorted(
        estaciones
    )


# ============================================================
# GENERAR PARES
# ============================================================

def generar_pares(
    grafo,
    cantidad
):

    estaciones = (
        obtener_estaciones(grafo)
    )

    print(
        f"Estaciones/nodos encontrados: "
        f"{len(estaciones)}"
    )

    if len(estaciones) < 2:

        raise RuntimeError(
            "No hay suficientes estaciones."
        )

    random.seed(
        SEMILLA
    )

    pares = set()

    intentos = 0

    max_intentos = (
        cantidad * 20
    )

    while (
        len(pares) < cantidad
        and intentos < max_intentos
    ):

        origen, destino = random.sample(
            estaciones,
            2
        )

        pares.add(
            (
                origen,
                destino
            )
        )

        intentos += 1

    return list(
        pares
    )


# ============================================================
# CREAR DATASET
# ============================================================

def generar_dataset():

    print("=" * 60)
    print(
        " GENERADOR DATASET ML - TRANSMI ROUTE"
    )
    print("=" * 60)
    print()

    rutas = cargar_datos()

    grafo, estaciones = (
        construir_grafo(rutas)
    )

    grafo = limpiar_grafo(
        grafo
    )

    print(
        f"Estaciones: {len(estaciones)}"
    )

    print(
        f"Nodos del grafo: {len(grafo)}"
    )

    print()

    pares = generar_pares(
        grafo,
        MAX_PARES
    )

    registros = []

    id_recorrido = 1

    pares_con_alternativas = 0

    total_rutas = 0

    for origen, destino in pares:

        rutas_candidatas = (
            generar_rutas_alternativas(
                grafo,
                origen,
                destino,
                MAX_RUTAS_POR_PAR
            )
        )

        if not rutas_candidatas:
            continue

        if len(rutas_candidatas) < 2:
            continue

        pares_con_alternativas += 1

        evaluadas = []

        for ruta in rutas_candidatas:

            caracteristicas = (
                calcular_caracteristicas(
                    ruta
                )
            )

            evaluadas.append(
                (
                    ruta,
                    caracteristicas
                )
            )

        # ----------------------------------------------------
        # Ordenamos según las reglas del proyecto
        #
        # 1. Menos transbordos
        # 2. Menos estaciones
        # 3. Menos servicios
        # ----------------------------------------------------

        evaluadas.sort(
            key=lambda elemento:
                clave_evaluacion(
                    elemento[1]
                )
        )

        mejor_clave = (
            clave_evaluacion(
                evaluadas[0][1]
            )
        )

        for ruta, caracteristicas in (
            evaluadas
        ):

            clave = (
                clave_evaluacion(
                    caracteristicas
                )
            )

            if clave == mejor_clave:

                clasificacion = (
                    "RECOMENDADA"
                )

            else:

                clasificacion = (
                    "NO_RECOMENDADA"
                )

            registros.append(
                {
                    "id_recorrido":
                        id_recorrido,

                    "origen":
                        origen,

                    "destino":
                        destino,

                    "cantidad_estaciones":
                        caracteristicas[
                            "cantidad_estaciones"
                        ],

                    "cantidad_tramos":
                        caracteristicas[
                            "cantidad_tramos"
                        ],

                    "cantidad_servicios":
                        caracteristicas[
                            "cantidad_servicios"
                        ],

                    "cantidad_transbordos":
                        caracteristicas[
                            "cantidad_transbordos"
                        ],

                    "clasificacion":
                        clasificacion,
                }
            )

            id_recorrido += 1
            total_rutas += 1

    if not registros:

        raise RuntimeError(
            "No se pudieron generar "
            "rutas alternativas."
        )

    # ========================================================
    # GUARDAR CSV
    # ========================================================

    campos = [
        "id_recorrido",
        "origen",
        "destino",
        "cantidad_estaciones",
        "cantidad_tramos",
        "cantidad_servicios",
        "cantidad_transbordos",
        "clasificacion",
    ]

    ARCHIVO_SALIDA.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        ARCHIVO_SALIDA,
        "w",
        newline="",
        encoding="utf-8"
    ) as archivo:

        escritor = csv.DictWriter(
            archivo,
            fieldnames=campos
        )

        escritor.writeheader()

        escritor.writerows(
            registros
        )

    # ========================================================
    # RESUMEN
    # ========================================================

    recomendadas = sum(
        1
        for registro in registros
        if registro[
            "clasificacion"
        ] == "RECOMENDADA"
    )

    no_recomendadas = sum(
        1
        for registro in registros
        if registro[
            "clasificacion"
        ] == "NO_RECOMENDADA"
    )

    print()
    print("=" * 60)
    print(" DATASET ML GENERADO")
    print("=" * 60)

    print(
        f"Pares con rutas alternativas: "
        f"{pares_con_alternativas}"
    )

    print(
        f"Total de recorridos: "
        f"{total_rutas}"
    )

    print(
        f"RECOMENDADA: "
        f"{recomendadas}"
    )

    print(
        f"NO_RECOMENDADA: "
        f"{no_recomendadas}"
    )

    print()
    print(
        f"Archivo:"
    )

    print(
        ARCHIVO_SALIDA
    )

    print()
    print(
        "Criterios de clasificación:"
    )

    print(
        "1. Menos transbordos"
    )

    print(
        "2. Menos estaciones"
    )

    print(
        "3. Menos servicios"
    )


# ============================================================
# EJECUCIÓN
# ============================================================

if __name__ == "__main__":

    generar_dataset()