import importlib.util
import unittest
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]


def _cargar_modulo(nombre_archivo):
    ruta = BASE_DIR / "src" / "machine_learning" / nombre_archivo
    spec = importlib.util.spec_from_file_location(nombre_archivo, ruta)
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


class TestRutasML(unittest.TestCase):
    def test_entrenar_arbol_usa_root_correcto(self):
        modulo = _cargar_modulo("entrenar_arbol.py")

        self.assertEqual(modulo.ROOT_DIR, BASE_DIR)
        self.assertTrue(modulo.ARCHIVO_DATASET.exists())

    def test_validar_dataset_usa_root_correcto(self):
        modulo = _cargar_modulo("validar_dataset.py")

        self.assertEqual(modulo.ROOT_DIR, BASE_DIR)
        self.assertTrue(modulo.ARCHIVO.exists())


if __name__ == "__main__":
    unittest.main()
