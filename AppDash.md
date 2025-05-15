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


```python
import dash
from dash import dcc, html, Input, Output, dash_table, State # State fue importado implicitamente antes, ahora explícito
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import joblib
from RenameDatabase import renameDatabase # Asumiendo que este script existe y es importable

import warnings
warnings.filterwarnings('ignore') # Ignorar advertencias (generalmente no recomendado para producción)

# --- Carga de Datos ---
# Cargar el conjunto de datos principal de estadísticas de jugadores de la NBA
df = pd.read_csv('2023_nba_player_stats.csv')
# Renombrar columnas o realizar otras transformaciones usando el script externo
df = renameDatabase(df)

# Cargar datos adicionales: 'database.csv' podría ser para el modelo, 'results_df' para comparación de modelos
model_data = pd.read_csv('database.csv')
results_df = pd.read_csv('Model comparison results.csv')

# --- Carga de Modelos Entrenados ---
try:
    # Cargar los modelos de machine learning pre-entrenados
    trained_models = joblib.load('models/top_three_models.pkl')
    # Cargar la lista de características (features) que los modelos esperan
    model_features = joblib.load('models/model_features.pkl')
    print("Modelos cargados correctamente:")
    for name in trained_models.keys():
        print(f"- {name}")
except Exception as e:
    print(f"Error al cargar los modelos: {e}")
    trained_models = {} # Si hay error, inicializar como diccionario vacío
    model_features = [] # Si hay error, inicializar como lista vacía

# --- Inicialización de la Aplicación Dash ---
# Crear una instancia de la aplicación Dash.
# external_stylesheets se usa para aplicar estilos CSS, aquí se usa un tema de Bootstrap.
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server # Exponer el servidor Flask subyacente (necesario para algunas plataformas de despliegue)

# --- Definición del Layout de la Aplicación ---
# El layout describe cómo se verá la aplicación. Se usan componentes HTML y Dash Core Components (dcc).
app.layout = dbc.Container([
    # Título principal del dashboard
    html.H1("NBA Player Stats Analysis Dashboard",
            style={'textAlign': 'center', 'color': '#1D428A', 'marginBottom':30, 'marginTop':20}),

    # Componente de Pestañas (Tabs) para organizar el contenido
    dcc.Tabs([
        # Pestaña 1: Vista General de Datos
        dcc.Tab(label='Data Overview', children=[
            dbc.Row([ # Fila para organizar contenido
                dbc.Col([ # Columna dentro de la fila
                    html.H3("NBA Player Stats Overview", style={'margin': '20px 0px'}),
                    # Sección para mostrar estadísticas descriptivas del DataFrame principal
                    html.Div([
                        html.H4("Dataset Statistics"),
                        dash_table.DataTable(
                            data = df.describe().reset_index().round(2).to_dict('records'), # Datos para la tabla
                            columns = [{"name": i, "id": i} for i in df.describe().reset_index().columns], # Definición de columnas
                            style_table = {'overflowX': 'auto'}, # Estilo para la tabla (scroll horizontal)
                            style_cell = {'textAlign': 'center', 'padding': '10px'}, # Estilo para celdas
                            style_header = {'backgroundColor': '#1D428A', 'color': 'white', 'fontWeight': 'bold'} # Estilo para encabezado
                        )
                    ], style={'width': '100%', 'margin': '10px'}),
                    # Sección para mostrar una muestra de los datos (primeras 10 filas)
                    html.Div([
                        html.H4("Data Sample"),
                        dash_table.DataTable(
                            data = df.head(10).to_dict('records'),
                            columns = [{"name": i, "id": i} for i in df.columns],
                            style_table = {'overflowX': 'auto', 'maxHeight': '400px'},
                            style_cell = {'textAlign': 'center', 'padding': '5px'},
                            style_header = {'backgroundColor': '#1D428A', 'color': 'white', 'fontWeight': 'bold'}
                        )
                    ], style={'width': '100%', 'margin': '10px'}),
                ])
            ])
        ]),

        # Pestaña 2: Análisis de Estadísticas del Jugador
        dcc.Tab(label='Player Stats Analysis', children=[
            dbc.Row([
                dbc.Col([
                    html.H3("Player Stats Analysis", style={'margin': '20px 0px'}),
                    # Fila para los primeros dos gráficos
                    dbc.Row([
                        dbc.Col([
                            html.H4("Points Distribution"),
                            # Gráfico de histograma para la distribución de puntos totales
                            dcc.Graph(figure = px.histogram(df, x = 'Total_Points',
                                                            title = 'Distribution of Total Points',
                                                            color_discrete_sequence = ['#C8102E']))
                        ], width=6), # Ancho de la columna (sistema de 12 columnas de Bootstrap)
                        dbc.Col([
                            html.H4("Points per Position"),
                            # Gráfico de caja para la distribución de puntos por posición
                            dcc.Graph(figure = px.box(df, x ='Position', y = 'Total_Points',
                                                      title = 'Points Distribution by Position',
                                                      color = 'Position'))
                        ], width=6),
                    ]),
                    # Fila para los siguientes dos gráficos
                    dbc.Row([
                        dbc.Col([
                            html.H4("Points vs Age"),
                            # Gráfico de dispersión de Edad vs Puntos Totales, con línea de tendencia
                            dcc.Graph(figure = px.scatter(df, x = 'Age', y = 'Total_Points',
                                                          title = 'Age vs Total Points',
                                                          color = 'Position',
                                                          trendline = 'ols')) # 'ols' para Ordinary Least Squares trendline
                        ], width=6),
                        dbc.Col([
                            html.H4("Correlation Heatmap"),
                            # Mapa de calor de la matriz de correlación de 'model_data'
                            dcc.Graph(figure = px.imshow(model_data.corr(numeric_only=True), # numeric_only=True para evitar errores con no numéricos
                                                         title = 'Feature Correlation Matrix',
                                                         color_continuous_scale = 'RdBu_r'))
                        ], width=6),
                    ]),
                    # Fila para el explorador de características
                    dbc.Row([
                        dbc.Col([
                            html.H4("Feature Explorer"),
                            html.P("Select a feature to analyze its relationship with Total Points:"),
                            # Menú desplegable para seleccionar una característica
                            dcc.Dropdown(
                                id = 'feature-dropdown', # ID para usar en callbacks
                                options = [{'label': col, 'value': col} for col in model_data.columns if col != 'Total_Points'], # Opciones del dropdown
                                value = 'Field_Goals_Attempted', # Valor por defecto
                                style = {'width': '50%'}),
                            # Gráfico que se actualizará interactivamente mediante un callback
                            dcc.Graph(id='feature-points-scatter')
                        ])
                    ])
                ])
            ])
        ]),

        # Pestaña 3: Rendimiento del Modelo
        dcc.Tab(label='Model Performance', children=[
            dbc.Row([
                dbc.Col([
                    html.H3("Model Performance Analysis", style={'margin': '20px 0px'}),
                    # Fila para gráficos de comparación de errores y R²
                    dbc.Row([
                        dbc.Col([
                            html.H4("Error Metrics Comparison"),
                            # Gráfico de barras comparando Test_MSE y Test_MAE por modelo
                            dcc.Graph(figure = px.bar(results_df, x = 'Model', y = ['Test_MSE', 'Test_MAE'],
                                                      barmode = 'group', # Barras agrupadas
                                                      title = 'Error Metrics by Model',
                                                      color_discrete_sequence = ['#C8102E', '#1D428A']))
                        ], width=6),
                        dbc.Col([
                            html.H4("R² Score by Model"),
                            # Gráfico de barras mostrando la puntuación R² por modelo
                            dcc.Graph(figure = px.bar(results_df, x = 'Model', y = 'Test_R2',
                                                      title = 'R² Score by Model',
                                                      color = 'Test_R2',
                                                      color_continuous_scale = 'viridis'))
                        ], width=6),
                    ]),
                    # Fila para el gráfico de rendimiento vs tiempo de entrenamiento
                    dbc.Row([
                        dbc.Col([
                            html.H4("Performance vs Training Time"),
                            # Gráfico de dispersión de Tiempo de Entrenamiento vs MAE, tamaño por MSE
                            dcc.Graph(figure = px.scatter(results_df, x = 'Training_Time', y = 'Test_MAE',
                                                          size = 'Test_MSE', size_max = 50,
                                                          hover_name = 'Model', # Información al pasar el mouse
                                                          title = 'Model Performance vs Training Time',
                                                          labels = {'Training_Time': 'Training Time (s)', 'Test_MAE': 'Absolute Error'},
                                                          color = 'Model'))
                        ])
                    ]),
                    # Fila para la tabla detallada de métricas del modelo
                    dbc.Row([
                        dbc.Col([
                            html.H4("Model Metrics Details"),
                            dash_table.DataTable(
                                data = results_df.round(3).to_dict('records'),
                                columns = [{"name": i, "id": i} for i in results_df.columns],
                                style_table = {'overflowX': 'auto'},
                                style_cell = {'textAlign': 'center', 'padding': '10px'},
                                style_header = {'backgroundColor': '#1D428A', 'color': 'white', 'fontWeight': 'bold'},
                                sort_action = 'native', # Habilitar ordenamiento nativo
                                filter_action = 'native' # Habilitar filtrado nativo
                            )
                        ])
                    ])
                ])
            ])
        ]),

        # Pestaña 4: Predecir Puntos
        dcc.Tab(label = 'Predict Points', children=[
            dbc.Row([
                dbc.Col([
                    html.H3("Player Points Predictor", style={'margin': '20px 0px'}),
                    # Tarjeta para agrupar los inputs de predicción
                    dbc.Card([
                        dbc.CardHeader(html.H4("Input Player Stats")), # Encabezado de la tarjeta
                        dbc.CardBody([ # Cuerpo de la tarjeta
                            # Varias filas de inputs para las estadísticas del jugador
                            dbc.Row([
                                dbc.Col([
                                    html.Label("Minutes Played: "),
                                    dcc.Input(id='minutes-input', type='number', value=0, style={'width': '100%'})
                                ], width=4),
                                dbc.Col([
                                    html.Label("Field Goals Attempted: "),
                                    dcc.Input(id='fga-input', type='number', value=0, style={'width': '100%'})
                                ], width=4),
                                dbc.Col([
                                    html.Label("Three Point Attempts: "),
                                    dcc.Input(id='3pa-input', type='number', value=0, style={'width': '100%'})
                                ], width=4)
                            ], className = "mb-3"), # Margen inferior
                            dbc.Row([
                                dbc.Col([
                                    html.Label("Free Throws Attempted: "),
                                    dcc.Input(id='fta-input', type='number', value=0, style={'width': '100%'})
                                ], width=4),
                                dbc.Col([
                                    html.Label("Assists: "),
                                    dcc.Input(id='ast-input', type='number', value=0, style={'width': '100%'})
                                ], width=4),
                                dbc.Col([
                                    html.Label("Steals: "),
                                    dcc.Input(id='stl-input', type='number', value=0, style={'width': '100%'})
                                ], width=4)
                            ], className = "mb-3"),
                            dbc.Row([
                                dbc.Col([
                                    html.Label("Turnovers: "),
                                    dcc.Input(id='tov-input', type='number', value=0, style={'width': '100%'})
                                ], width=4),
                                dbc.Col([
                                    html.Label("Total Rebounds: "),
                                    dcc.Input(id='reb-input', type='number', value=0, style={'width': '100%'})
                                ], width=4),
                                dbc.Col([
                                    html.Label("Personal Fouls: "),
                                    dcc.Input(id='pf-input', type='number', value=0, style={'width': '100%'})
                                ], width=4)
                            ], className = "mb-3"),
                            # Botón para activar la predicción
                            dbc.Button('Predict Points', id = 'predict-button',
                                       color = 'danger', className = 'mt-3'), # Color del botón y margen superior
                            # Div para mostrar el resultado de la predicción textual
                            html.Div(id = 'prediction-output',
                                     style = {'margin': '20px 0', 'fontWeight': 'bold', 'fontSize': '18px'}),
                        ])
                    ], style = {'marginBottom': '20px'}),
                    # Div para el gráfico de comparación de predicciones
                    html.Div([
                        html.H4("Prediction Comparison"),
                        dcc.Graph(id='prediction-comparison') # Gráfico actualizado por callback
                    ])
                ])
            ])
        ])
    ], style={'marginTop': '20px'}) # Margen superior para las pestañas
], fluid=True) # fluid=True hace que el contenedor ocupe todo el ancho disponible

# --- Definición de Callbacks (Interactividad) ---

# Callback 1: Actualizar el gráfico del explorador de características
@app.callback(
    Output('feature-points-scatter', 'figure'), # El output es la propiedad 'figure' del gráfico con id 'feature-points-scatter'
    [Input('feature-dropdown', 'value')] # El input es la propiedad 'value' del dropdown con id 'feature-dropdown'
)
def update_features_scatter(feature_selected):
    """
    Actualiza el gráfico de dispersión en la pestaña 'Player Stats Analysis'
    basado en la característica seleccionada en el menú desplegable.
    """
    # Crear un gráfico de dispersión con Plotly Express
    fig = px.scatter(df, x = feature_selected, y = 'Total_Points', # Usar el DataFrame principal 'df'
                     color = 'Position', # Colorear puntos por posición del jugador
                     title = f'{feature_selected} vs Total Points', # Título dinámico
                     trendline = 'ols') # Añadir línea de tendencia (Ordinary Least Squares)
    return fig # Retornar la figura del gráfico

# Callback 2: Predecir puntos y actualizar el gráfico de comparación de predicciones
@app.callback(
    [Output('prediction-output', 'children'), # Output 1: El contenido (texto) del div 'prediction-output'
     Output('prediction-comparison', 'figure')], # Output 2: La figura del gráfico 'prediction-comparison'
    [Input('predict-button', 'n_clicks')], # Input: El número de clics del botón 'predict-button'
    [State('minutes-input', 'value'), # State: Captura el valor de los inputs sin disparar el callback
     State('fga-input', 'value'),
     State('3pa-input', 'value'),
     State('fta-input', 'value'),
     State('ast-input', 'value'),
     State('stl-input', 'value'),
     State('tov-input', 'value'),
     State('reb-input', 'value'),
     State('pf-input', 'value')]
)
def predict_points(n_clicks, minutes, fga, tpa, fta, ast, stl, tov, reb, pf):
    """
    Realiza predicciones de puntos basadas en las estadísticas ingresadas por el usuario
    y actualiza el texto de predicción y el gráfico de comparación.
    """
    # Si el botón no ha sido presionado, mostrar mensaje de espera y figura vacía
    if n_clicks is None or n_clicks == 0:
        return "Esperando datos de entrada...", go.Figure() # go.Figure() crea una figura vacía

    # Verificar si los modelos y características fueron cargados correctamente
    if not trained_models or not model_features:
        return "Error al cargar los datos del modelo.", go.Figure()

    # Recopilar los valores de entrada
    input_values = [minutes, fga, tpa, fta, ast, stl, tov, reb, pf]
    # Verificar si alguno de los campos está vacío (None)
    if any(x is None for x in input_values):
        return "Por favor, ingrese todos los campos.", go.Figure()

    # Crear un diccionario con los datos de entrada para el DataFrame
    # Los nombres de las claves deben coincidir con los nombres de las características esperadas por los modelos
    input_data_dict = {
        'Minutes_Played': [minutes],
        'Field_Goals_Attempted': [fga],
        'Three_Point_FG_Attempted': [tpa],
        'Free_Throws_Attempted': [fta],
        'Total_Rebounds': [reb],
        'Assists': [ast],
        'Turnovers': [tov],
        'Steals': [stl],
        'Personal_Fouls': [pf]
    }
    # Convertir el diccionario a un DataFrame de Pandas
    input_data = pd.DataFrame(input_data_dict)

    # Asegurar que el DataFrame de entrada tenga todas las características que los modelos esperan,
    # en el orden correcto, y rellenar con 0 las que falten.
    for col in model_features:
        if col not in input_data.columns:
            input_data[col] = 0 # Añadir columnas faltantes con valor 0
    input_data = input_data.reindex(columns=model_features, fill_value=0) # Reordenar y rellenar

    # Realizar predicciones con cada modelo cargado
    predictions = {} # Diccionario para almacenar las predicciones de cada modelo
    for name, model in trained_models.items():
        try:
            pred = model.predict(input_data)[0] # Obtener la predicción (usualmente un array, tomar el primer elemento)
            predictions[name] = max(0, pred) # Asegurar que la predicción no sea negativa
        except Exception as e:
            print(f"Fallo al predecir con el modelo {name}: {e}")
            predictions[name] = 0 # Asignar 0 si hay un error en la predicción

    # Crear el gráfico de barras para comparar las predicciones
    fig_comparison = go.Figure()
    for model_name, pred_value in predictions.items():
        fig_comparison.add_trace(go.Bar(
            x = [model_name], # Nombre del modelo en el eje x
            y = [pred_value], # Valor de la predicción en el eje y
            name = model_name, # Nombre para la leyenda
        ))
    fig_comparison.update_layout(
        title_text = "Puntos Predichos por Diferentes Modelos", # Título del gráfico
        yaxis_title_text = 'Puntos Predichos', # Título del eje y
        barmode = 'group' # Modo de las barras (agrupadas)
    )

    # Preparar el texto del resultado
    if predictions: # Si se realizaron predicciones
        # Calcular el promedio de las predicciones
        avg_prediction = round(sum(predictions.values()) / len(predictions), 1)
        result_text = f"Puntos Predichos (promedio): {avg_prediction}"
    else: # Si no se pudieron realizar predicciones
        result_text = "Fallo al realizar las predicciones."
    
    return result_text, fig_comparison # Retornar el texto y la figura del gráfico

# --- Ejecución de la Aplicación ---
# Este bloque se ejecuta solo si el script es corrido directamente (no importado como módulo)
if __name__ == '__main__':
    # Iniciar el servidor de desarrollo de Dash
    # debug=True habilita el modo de depuración (útil para desarrollo, muestra errores en el navegador)
    # port=8050 especifica el puerto en el que correrá la aplicación
    app.run(debug=True, port=8050)
```

