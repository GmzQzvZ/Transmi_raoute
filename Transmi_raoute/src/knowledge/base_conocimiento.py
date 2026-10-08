from pathlib import Path

from src.utils.helpers import cargar_json
from src.utils.helpers import limpiar_nombre_estacion


RUTA_DATOS_ORIENTADOS = (
    Path(__file__).resolve().parents[2] / "data" / "processed" / "rutas_orientadas.json"
)

ALIAS_ESTACIONES_VALIDADOS = {
    "portal del sur": "portal sur jfk coop. financiera",
    "portal de la 80": "portal 80",
    "portal del norte": "portal norte",
    "toberin": "toberin foundever",
    "las aguas": "las aguas centro colombo americano",
    "universidades": "universidades - cityu",
    "san mateo": "san mateo c.c. unisur",
    "museo nacional": "museo nacional fng",
    "flores": "flores areandina",
    "alcala": "alcala - colegio s. tomas dominicos",
    "calle 76": "calle 76 - san felipe",
    "portal el dorado": 116,
    "portal eldorado": 116,
}

def cargar_rutas_orientadas(ruta_archivo=None):
    """Devuelve las rutas orientadas ya procesadas y validadas."""
    return cargar_json(ruta_archivo or RUTA_DATOS_ORIENTADOS)


def normalizar_aliases(alias_estaciones=None):
    aliases = alias_estaciones or ALIAS_ESTACIONES_VALIDADOS
    aliases_normalizados = {}
    for alias, destino in aliases.items():
        alias_normalizado = limpiar_nombre_estacion(alias)
        if isinstance(destino, int):
            aliases_normalizados[alias_normalizado] = destino
        else:
            aliases_normalizados[alias_normalizado] = limpiar_nombre_estacion(destino)
    return aliases_normalizados


def resolver_estacion_con_alias(grafo, estacion, alias_estaciones=None):
    """Resuelve una estacion por ID, nombre oficial o alias validado."""
    estacion_id, estado = grafo.resolver_estacion_detalle(estacion)
    if estado != "no_existe" or isinstance(estacion, int):
        return estacion_id, estado

    texto = str(estacion).strip()
    if texto.isdigit():
        return estacion_id, estado

    destino_alias = normalizar_aliases(alias_estaciones).get(limpiar_nombre_estacion(texto))
    if destino_alias is None:
        return estacion_id, estado

    if isinstance(destino_alias, int):
        if destino_alias in grafo.estaciones:
            return destino_alias, "ok"
        return None, "no_existe"

    coincidencias = grafo.indice_nombres.get(destino_alias, set())
    if len(coincidencias) == 1:
        return next(iter(coincidencias)), "ok"
    if len(coincidencias) > 1:
        return None, "ambiguo"
    return None, "no_existe"
