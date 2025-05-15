# Documentación: Dashboard de Análisis de Estadísticas de Jugadores de la NBA (Dash)

Este documento detalla la estructura y funcionamiento de la aplicación Dash desarrollada para analizar y predecir estadísticas de jugadores de la NBA, basada en el archivo `dash_app.py`.

## 1. Visión General

El dashboard permite a los usuarios explorar datos de jugadores de la NBA de la temporada 2023, analizar visualmente diferentes estadísticas, comparar el rendimiento de modelos de predicción de puntos y realizar predicciones de puntos para jugadores basadas en estadísticas de entrada.

La aplicación está estructurada en varias pestañas para una navegación intuitiva:
* **Data Overview:** Muestra estadísticas descriptivas y una muestra de los datos.
* **Player Stats Analysis:** Presenta análisis visuales como distribuciones de puntos, comparaciones por posición, correlaciones y un explorador de características.
* **Model Performance:** Compara el rendimiento de diferentes modelos de machine learning.
* **Predict Points:** Permite al usuario ingresar estadísticas de un jugador para predecir sus puntos.

## 2. Estructura del Archivo `dash_app.py`

El script sigue una estructura típica de una aplicación Dash:

1.  **Importaciones:** Se importan las bibliotecas necesarias (`dash`, `dcc`, `html`, `Input`, `Output`, `dash_table`, `dash_bootstrap_components`, `plotly.express`, `plotly.graph_objects`, `pandas`, `joblib`, `numpy` y un módulo local `RenameDatabase`).
2.  **Carga de Datos y Modelos:**
    * Se carga el dataset principal `2023_nba_player_stats.csv` y se procesa con `RenameDatabase.renameDatabase()`.
    * Se cargan datos adicionales `database.csv` (posiblemente para modelado) y `Model comparison results.csv`.
    * Se intentan cargar los modelos pre-entrenados (`top_three_models.pkl`) y las características del modelo (`model_features.pkl`) usando `joblib`. Se manejan excepciones si los archivos no se encuentran.
3.  **Inicialización de la App Dash:**
    * `app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])` crea la instancia de la aplicación, utilizando hojas de estilo de Bootstrap para un diseño base.
4.  **Definición del Layout (`app.layout`):**
    * Se utiliza `dbc.Container` como el contenedor principal.
    * Un título principal `html.H1`.
    * Un componente `dcc.Tabs` para organizar el contenido en las cuatro pestañas mencionadas anteriormente.
    * Dentro de cada `dcc.Tab`, se utilizan componentes `dbc.Row` y `dbc.Col` para estructurar el contenido en un sistema de rejilla.
    * Se emplean diversos componentes de Dash y HTML:
        * `html.H3`, `html.H4`, `html.P`, `html.Label` para texto.
        * `dash_table.DataTable` para mostrar tablas de datos interactivas.
        * `dcc.Graph` para mostrar gráficos de Plotly.
        * `dcc.Dropdown` para selección interactiva de características.
        * `dcc.Input` para la entrada de datos numéricos en el predictor.
        * `dbc.Card`, `dbc.CardHeader`, `dbc.CardBody` para agrupar contenido visualmente.
        * `dbc.Button` para activar la predicción.
5.  **Callbacks (`@app.callback`):**
    * Se definen dos callbacks principales para la interactividad.
6.  **Ejecución de la App:**
    * `if __name__ == '__main__': app.run(debug=True, port=8050)` inicia el servidor de desarrollo.

## 3. Componentes Detallados

### 3.1. Carga de Datos y Modelos

* **`df = pd.read_csv('2023_nba_player_stats.csv')`**: Carga las estadísticas crudas de los jugadores.
* **`df = renameDatabase(df)`**: Aplica una función (presumiblemente para limpiar o estandarizar nombres de columnas) al DataFrame principal.
* **`model_data = pd.read_csv('database.csv')`**: Carga un conjunto de datos que parece ser el utilizado para entrenar o evaluar los modelos.
* **`results_df = pd.read_csv('Model comparison results.csv')`**: Carga un CSV con los resultados de la comparación de diferentes modelos de ML.
* **`trained_models = joblib.load('models/top_three_models.pkl')`**: Carga un diccionario o lista de modelos de scikit-learn pre-entrenados.
* **`model_features = joblib.load('models/model_features.pkl')`**: Carga una lista de las características (nombres de columnas) que los modelos esperan como entrada.

### 3.2. Layout

#### Pestaña: Data Overview
* Muestra estadísticas descriptivas del DataFrame `df` usando `df.describe()` en un `dash_table.DataTable`.
* Muestra las primeras 10 filas de `df` en otro `dash_table.DataTable`.

#### Pestaña: Player Stats Analysis
* **Distribución de Puntos:** Histograma (`px.histogram`) de `Total_Points`.
* **Puntos por Posición:** Diagrama de caja (`px.box`) de `Total_Points` agrupado por `Position`.
* **Puntos vs Edad:** Gráfico de dispersión (`px.scatter`) de `Age` vs `Total_Points`, con línea de tendencia.
* **Mapa de Calor de Correlación:** Matriz de correlación (`px.imshow`) del `model_data`.
* **Explorador de Características:**
    * Un `dcc.Dropdown` (`id='feature-dropdown'`) permite seleccionar una característica del `model_data`.
    * Un `dcc.Graph` (`id='feature-points-scatter'`) muestra un gráfico de dispersión de la característica seleccionada contra `Total_Points`. Este gráfico se actualiza mediante un callback.

#### Pestaña: Model Performance
* **Comparación de Métricas de Error:** Gráfico de barras (`px.bar`) mostrando `Test_MSE` y `Test_MAE` para cada modelo de `results_df`.
* **Puntuación R² por Modelo:** Gráfico de barras (`px.bar`) mostrando `Test_R2` para cada modelo.
* **Rendimiento vs Tiempo de Entrenamiento:** Gráfico de dispersión (`px.scatter`) de `Training_Time` vs `Test_MAE`, con el tamaño de los puntos representando `Test_MSE`.
* **Detalles de Métricas del Modelo:** Un `dash_table.DataTable` muestra `results_df` completo.

#### Pestaña: Predict Points
* Un `dbc.Card` contiene los campos de entrada (`dcc.Input`) para varias estadísticas del jugador (Minutos Jugados, Tiros de Campo Intentados, etc.). Los IDs de estos inputs son `minutes-input`, `fga-input`, etc.
* Un `dbc.Button` (`id='predict-button'`) para iniciar la predicción.
* Un `html.Div` (`id='prediction-output'`) para mostrar el resultado de la predicción textual.
* Un `dcc.Graph` (`id='prediction-comparison'`) para mostrar un gráfico de barras comparando las predicciones de los diferentes modelos cargados.

### 3.3. Callbacks

#### Callback 1: Actualizar Gráfico del Explorador de Características
* **Input:** `Input('feature-dropdown', 'value')` (la característica seleccionada).
* **Output:** `Output('feature-points-scatter', 'figure')` (la figura del gráfico de dispersión).
* **Lógica:**
    1.  Recibe la característica seleccionada.
    2.  Genera un gráfico de dispersión (`px.scatter`) usando `df`, mostrando la `feature` seleccionada en el eje x y `Total_Points` en el eje y, coloreado por `Position` y con una línea de tendencia.
    3.  Retorna la figura generada.

#### Callback 2: Predecir Puntos y Actualizar Gráfico de Comparación
* **Input:** `Input('predict-button', 'n_clicks')` (el número de veces que se ha hecho clic en el botón).
* **States:** `State('minutes-input', 'value')`, `State('fga-input', 'value')`, ..., `State('pf-input', 'value')` (los valores de todos los campos de entrada de estadísticas del jugador).
* **Outputs:**
    * `Output('prediction-output', 'children')` (el texto con la predicción promedio).
    * `Output('prediction-comparison', 'figure')` (el gráfico de barras con las predicciones de cada modelo).
* **Lógica:**
    1.  Se activa cuando se hace clic en el botón `predict-button`.
    2.  Verifica si el botón ha sido presionado y si los modelos y características están cargados.
    3.  Recoge los valores de todos los `dcc.Input` usando `State`.
    4.  Crea un DataFrame de Pandas (`input_data`) con los valores ingresados, asegurándose de que los nombres de las columnas coincidan con los esperados por los modelos (`model_features`). Las columnas faltantes se rellenan con 0.
    5.  Itera sobre cada modelo en `trained_models`:
        * Realiza una predicción usando `model.predict(input_data)`.
        * Almacena la predicción (asegurándose que no sea negativa).
        * Maneja excepciones durante la predicción.
    6.  Crea un gráfico de barras (`go.Figure` con `go.Bar`) que muestra la predicción de cada modelo.
    7.  Calcula la predicción promedio si hay predicciones disponibles.
    8.  Retorna el texto del resultado (predicción promedio o mensaje de error) y la figura del gráfico de barras.

## 4. Ejecución de la Aplicación

Para ejecutar este dashboard localmente:

1.  Asegúrate de tener todas las dependencias instaladas (ver `requirements.txt` si está disponible, o instalar `dash`, `pandas`, `plotly`, `dash-bootstrap-components`, `joblib`, `numpy`).
2.  Coloca los archivos de datos (`2023_nba_player_stats.csv`, `database.csv`, `Model comparison results.csv`) y la carpeta `models` (con `top_three_models.pkl` y `model_features.pkl`) en el mismo directorio que `dash_app.py`, o ajusta las rutas en el script. Asegúrate también de que `RenameDatabase.py` esté accesible.
3.  Abre una terminal en el directorio del proyecto.
4.  Ejecuta el comando: `python dash_app.py`
5.  Abre tu navegador web y ve a `http://127.0.0.1:8050/`.

