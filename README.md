# 🚍 TransMi Route

Sistema inteligente desarrollado en Python para encontrar y evaluar rutas entre estaciones del sistema TransMilenio mediante **representación del conocimiento, grafos, reglas de decisión, búsqueda BFS y aprendizaje supervisado mediante árboles de decisión**.

## 📌 Descripción

**TransMi Route** representa la red troncal de TransMilenio como un grafo.

El sistema permite:

* Representar las estaciones como nodos.
* Representar las conexiones entre estaciones como aristas dirigidas.
* Utilizar **BFS (Breadth-First Search)** para encontrar recorridos.
* Identificar los servicios utilizados.
* Detectar transbordos.
* Calcular métricas de cada recorrido.
* Aplicar reglas para seleccionar una ruta.
* Generar un conjunto de datos para aprendizaje supervisado.
* Entrenar un **árbol de decisión**.
* Evaluar el modelo mediante métricas de clasificación.
* Presentar resultados mediante una interfaz gráfica desarrollada con Tkinter.

> **Nota:** la definición de "mejor ruta" utilizada por el proyecto corresponde a una política académica configurable. No representa una definición universal de la mejor ruta de TransMilenio.

---

# 🎯 Objetivos

## Objetivo general

Desarrollar un sistema inteligente en Python capaz de representar la red troncal de TransMilenio como un grafo, encontrar recorridos mediante BFS, evaluar las alternativas mediante reglas y utilizar aprendizaje supervisado para clasificar recorridos.

## Objetivos específicos

1. Construir una representación de la red de TransMilenio mediante grafos.
2. Implementar búsqueda BFS para encontrar recorridos.
3. Identificar servicios y transbordos.
4. Evaluar las características de los recorridos.
5. Construir un dataset para aprendizaje supervisado.
6. Entrenar un árbol de decisión.
7. Evaluar el desempeño del modelo.
8. Documentar las limitaciones y resultados del sistema.

---

# 🧠 Arquitectura general

```text
Datos de referencia
        │
        ▼
Procesamiento de datos
        │
        ▼
Rutas orientadas
        │
        ▼
Base de conocimiento
        │
        ▼
Construcción del grafo
        │
        ▼
       BFS
        │
        ▼
Evaluación de recorridos
        │
        ▼
Reglas de selección
        │
        ├───────────────────────┐
        │                       │
        ▼                       ▼
 Ruta recomendada       Dataset supervisado
                                │
                                ▼
                       Árbol de decisión
                                │
                                ▼
                           Predicción
                                │
                                ▼
                         Evaluación ML
```

---

# 📁 Estructura del proyecto

```text
Transmi_raoute/
│
├── data/
│   ├── raw/
│   │   ├── rutas_troncales_raw.json
│   │   ├── estaciones_troncales_raw.json
│   │   └── equivalencias_estaciones_rutas_raw.json
│   │
│   └── processed/
│       ├── rutas_transmi.json
│       ├── rutas_orientadas.json
│       └── dataset_transmilenio_ml.csv
│
├── src/
│   │
│   ├── algorithms/
│   │   └── bfs.py
│   │
│   ├── knowledge/
│   │   ├── base_conocimiento.py
│   │   └── reglas.py
│   │
│   ├── routing/
│   │   └── evaluador.py
│   │
│   ├── data_processing/
│   │
│   ├── utils/
│   │
│   └── machine_learning/
│       ├── generar_dataset_ml.py
│       ├── validar_dataset.py
│       └── entrenar_arbol.py
│
├── tests/
│   ├── test_bfs.py
│   ├── test_evaluador.py
│   ├── test_integracion.py
│   ├── test_main.py
│   └── test_reglas.py
│
├── ui/
│   ├── __init__.py
│   └── app.py
│
├── main.py
├── .gitignore
└── README.md
```

---

# 📊 Datos del proyecto

La versión validada del sistema cuenta con:

* **90 rutas**
* **99 direcciones**
* **99 direcciones construidas**
* **0 direcciones pendientes**
* **0 errores de orientación**
* **134 estaciones**
* **919 aristas dirigidas**
* **33 pruebas automatizadas**

Los datos de referencia provienen de información abierta relacionada con TransMilenio.

---

# 🚌 Representación como grafo

La red se representa mediante un grafo dirigido.

```text
Estación A
    │
    ▼
Estación B
    │
    ▼
Estación C
```

Cada estación representa un nodo y cada conexión entre estaciones consecutivas representa una arista.

Las aristas contienen información relacionada con el servicio que realiza el recorrido.

---

# 🔎 Algoritmo BFS

El sistema utiliza **Breadth-First Search (BFS)** para explorar el grafo.

De manera simplificada:

```text
Origen
  │
  ├── Estación 1
  │
  ├── Estación 2
  │
  └── Estación 3
          │
          ▼
       Destino
```

BFS permite encontrar un recorrido dentro de la red modelada y reconstruir las estaciones visitadas.

---

# 📏 Evaluación de rutas

Los recorridos se evalúan mediante características obtenidas directamente de la red:

* Cantidad de estaciones.
* Cantidad de tramos.
* Cantidad de servicios.
* Cantidad de transbordos.

La política inicial utiliza:

1. Menor cantidad de transbordos.
2. Menor cantidad de estaciones.
3. Menor cantidad de servicios.

La política es configurable y se encuentra relacionada con las reglas del sistema.

---

# 🤖 Machine Learning

Como parte de la actividad de aprendizaje supervisado, se construyó un dataset utilizando recorridos generados sobre la red.

El flujo es:

```text
Rutas orientadas
       │
       ▼
Construcción del grafo
       │
       ▼
Generación de recorridos
       │
       ▼
Extracción de características
       │
       ▼
Etiquetado
       │
       ▼
dataset_transmilenio_ml.csv
       │
       ▼
Árbol de decisión
```

## Características utilizadas

El modelo utiliza:

```text
cantidad_estaciones
cantidad_servicios
cantidad_transbordos
```

La variable objetivo es:

```text
clasificacion
```

con dos clases:

```text
RECOMENDADA
NO_RECOMENDADA
```

Las etiquetas se generan utilizando los criterios de evaluación definidos previamente en el sistema.

> Esto significa que el modelo aprende a reproducir una política de selección previamente definida. No se afirma que el árbol haya descubierto por sí mismo cuál es la mejor ruta.

---

# 📦 Dataset de Machine Learning

El dataset contiene:

* **566 recorridos**
* **179 recorridos RECOMENDADOS**
* **387 recorridos NO_RECOMENDADOS**
* **0 conflictos de clasificación**
* **0 duplicados exactos**

Columnas:

```text
id_recorrido
origen
destino
cantidad_estaciones
cantidad_tramos
cantidad_servicios
cantidad_transbordos
clasificacion
```

---

# 🧪 Validación del dataset

El archivo:

```text
src/machine_learning/validar_dataset.py
```

comprueba:

* Existencia del dataset.
* Cantidad de registros.
* Columnas.
* Distribución de clases.
* Duplicados.
* Conflictos de clasificación.
* Estadísticas descriptivas.

Resultado validado actualmente:

```text
Total registros: 566

RECOMENDADA: 179
NO_RECOMENDADA: 387

Duplicados exactos: 0

Casos donde las mismas características
tienen clases diferentes: 0
```

---

# 🌳 Árbol de decisión

El modelo utilizado es:

```python
DecisionTreeClassifier(
    criterion="entropy",
    max_depth=4,
    random_state=42
)
```

El dataset se divide en:

```text
80 % entrenamiento
20 % prueba
```

utilizando `random_state=42` y división estratificada.

El modelo permite analizar la relación entre las características del recorrido y su clasificación.

---

# 📈 Resultados del modelo

El entrenamiento genera archivos dentro de:

```text
data/processed/ml_resultados/
```

Entre ellos:

```text
arbol_transmi_route.joblib
arbol_decision.png
matriz_confusion.png
importancia_caracteristicas.png
importancia_caracteristicas.csv
predicciones_prueba.csv
```

Estos archivos permiten analizar:

* El árbol entrenado.
* La matriz de confusión.
* La importancia de las características.
* Las predicciones realizadas sobre los datos de prueba.

---

# 🖥️ Interfaz gráfica

La interfaz utiliza **Tkinter**.

Permite:

* Ingresar origen.
* Ingresar destino.
* Buscar una ruta.
* Visualizar estaciones.
* Consultar cantidad de estaciones.
* Consultar cantidad de tramos.
* Consultar servicios.
* Identificar transbordos.
* Informar cuando no existe una ruta.

---

# ▶️ Ejecución del proyecto

Desde la raíz:

```text
C:\Users\Usuario\Desktop\python\Transmi_raoute
```

## Interfaz gráfica

```bat
python -m ui.app
```

## Aplicación de consola

```bat
python main.py
```

---

# 🤖 Ejecución de Machine Learning

## 1. Generar dataset

```bat
python src\machine_learning\generar_dataset_ml.py
```

## 2. Validar dataset

```bat
python src\machine_learning\validar_dataset.py
```

## 3. Entrenar árbol

```bat
python src\machine_learning\entrenar_arbol.py
```

---

# 🧪 Pruebas automatizadas

Para ejecutar todas las pruebas:

```bat
python -m unittest discover -s tests -p "test_*.py"
```

Resultado esperado:

```text
33 pruebas
OK
```

---

# ⚠️ Limitaciones

Actualmente el proyecto:

* Utiliza datos procesados localmente.
* No utiliza tráfico en tiempo real.
* No calcula tiempos reales de llegada.
* No incorpora congestión en tiempo real.
* No incorpora todavía criterios avanzados de accesibilidad.
* El dataset de Machine Learning se deriva de la política de evaluación existente.
* El modelo no representa una predicción de tiempo real del sistema TransMilenio.

---

# 🚀 Posibles mejoras

Como trabajo futuro se plantea:

* Incorporar información en tiempo real.
* Incorporar tiempos estimados de viaje.
* Incorporar congestión.
* Utilizar información de horarios.
* Incorporar accesibilidad.
* Comparar diferentes algoritmos de Machine Learning.
* Comparar árbol de decisión con otros clasificadores.
* Visualizar las rutas sobre un mapa.
* Incorporar nuevas variables al dataset cuando exista una fuente confiable.

---

# 🛠️ Tecnologías

* Python
* Tkinter
* JSON
* Pandas
* Scikit-learn
* Matplotlib
* Joblib
* unittest
* Git
* GitHub
* Grafos
* BFS
* Árboles de decisión

---

# 📚 Propósito académico

El proyecto integra conceptos de:

* Inteligencia Artificial.
* Representación del conocimiento.
* Sistemas basados en reglas.
* Grafos.
* Algoritmos de búsqueda.
* Aprendizaje supervisado.
* Árboles de decisión.
* Evaluación de modelos.

---

# 👥 Equipo

* Julian David Sanchez
* Juan Sebastian Gomez

---

# 📌 Estado

**Estado: funcional para el alcance académico definido.**

El proyecto incluye representación de la red, construcción del grafo, búsqueda BFS, evaluación de rutas, reglas de selección, pruebas automatizadas, interfaz gráfica y un componente de aprendizaje supervisado basado en árbol de decisión.
