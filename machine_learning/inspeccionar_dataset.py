import json
from pathlib import Path

ARCHIVO = Path(__file__).resolve().parents[1] / "data" / "processed" / "rutas_orientadas.json"

print("=" * 60)
print("INSPECCIÓN DE rutas_orientadas.json")
print("=" * 60)
print(f"Archivo: {ARCHIVO}")

with open(ARCHIVO, "r", encoding="utf-8") as f:
    datos = json.load(f)

print("\nTipo principal:")
print(type(datos))

if isinstance(datos, dict):
    print("\nClaves principales:")
    for clave in datos.keys():
        print(f"  - {clave}")

    print("\nCantidad de elementos por clave:")
    for clave, valor in datos.items():
        try:
            print(f"  {clave}: {len(valor)}")
        except TypeError:
            print(f"  {clave}: {type(valor).__name__}")

    print("\nPrimeros datos:")

    for clave, valor in datos.items():
        print(f"\n--- {clave} ---")
        print(repr(valor)[:3000])

else:
    print(f"\nCantidad de elementos: {len(datos)}")
    print("\nPrimer elemento:")
    print(repr(datos[0])[:3000])