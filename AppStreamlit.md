# Documentación: Dashboard de Análisis de Estadísticas de Jugadores de la NBA (Streamlit)

Este documento detalla la estructura y funcionamiento de la aplicación Streamlit desarrollada para analizar y predecir estadísticas de jugadores de la NBA, basada en el archivo `streamlit_app.py`.
Puedes ver la app funcional en este [enlace](https://dashboardfortestingframeworks.streamlit.app/).

## 1. Visión General

El dashboard, similar a su contraparte Dash, permite a los usuarios interactuar con datos de jugadores de la NBA de la temporada 2023. Ofrece visualizaciones, análisis de rendimiento de modelos y una herramienta de predicción de puntos.

La interfaz de usuario está organizada en pestañas y utiliza un diseño responsivo gracias a las capacidades de Streamlit.

Funcionalidades principales:
* **Data Overview:** Resumen estadístico, forma del dataset y distribución de posiciones.
* **Player Stats Analysis:** Gráficos interactivos para explorar relaciones entre estadísticas de jugadores.
* **Model Performance:** Visualización comparativa del rendimiento de modelos de machine learning.
* **Predict Points:** Interfaz para que los usuarios ingresen datos y obtengan predicciones de puntos.

## 2. Estructura del Archivo `streamlit_app.py`

El script de Streamlit sigue un flujo más secuencial:

1.  **Importaciones:** Se importan `streamlit`, `pandas`, `plotly.express`, `joblib` y `warnings`.
2.  **Configuración de Página y CSS:**
    * `st.set_page_config()`: Establece el título de la página y el layout (`wide`).
    * `st.markdown()`: Inyecta CSS personalizado para estilizar la aplicación con colores temáticos de la NBA y mejorar la apariencia de encabezados y tarjetas.
3.  **Funciones de Carga de Datos y Modelos (con Caching):**
    * `load_data()`: Función decorada con `@st.cache_data` para cargar y procesar los archivos CSV (`2023_nba_player_stats.csv`, `database.csv`, `Model comparison results.csv`). Incluye la importación y uso de `RenameDatabase.renameDatabase()`. Maneja errores si los archivos no se encuentran.
    * `load_models()`: Función decorada con `@st.cache_resource` para cargar los modelos pre-entrenados (`top_three_models.pkl`) y las características del modelo (`model_features.pkl`) usando `joblib`. Maneja errores.
4.  **Título Principal y Carga de Datos/Modelos:**
    * Se muestra el título principal de la aplicación usando `st.markdown()`.
    * Se llaman a `load_data()` y `load_models()` para obtener los datos y modelos. Se realizan comprobaciones para detener la app si los archivos esenciales no se cargan.
5.  **Creación de Pestañas (`st.tabs`):**
    * Se crean cuatro pestañas: "Data Overview", "Player Stats Analysis", "Model Performance", y "Predict Points".
6.  **Contenido de Cada Pestaña:**
    * Dentro de cada bloque `with tabX:`, se define el contenido usando funciones de Streamlit.
    * Se utilizan `st.markdown()` para sub-encabezados, `st.columns()` para organizar contenido lado a lado.
    * Los elementos visuales como tablas (`st.dataframe`, `st.write` para forma) y gráficos (`st.plotly_chart`, `st.bar_chart`) se renderizan directamente.
    * Los widgets interactivos como `st.selectbox()` y `st.number_input()` se utilizan para la entrada del usuario.
    * La lógica de predicción se activa con un `st.button()`.
7.  **Pie de Página:**
    * Un `st.markdown()` añade un pie de página simple.

## 3. Componentes Detallados

### 3.1. Configuración y Estilo
* `st.set_page_config(page_title="NBA Player Stats Analysis", layout="wide")`: Optimiza el uso del espacio en pantalla.
* El bloque CSS personalizado mejora la estética general, definiendo estilos para `.main-header`, `.sub-header`, y `.card` (utilizado para envolver secciones de contenido).

### 3.2. Carga de Datos y Modelos con Caching
* **`@st.cache_data` (para `load_data`)**: Asegura que los datos CSV solo se carguen y procesen una vez, incluso si el script se re-ejecuta debido a interacciones del usuario. Esto mejora significativamente el rendimiento.
* **`@st.cache_resource` (para `load_models`)**: Similar a `cache_data`, pero para recursos que no son fácilmente serializables como los modelos de `joblib`. Evita recargar los modelos en cada interacción.
* Las funciones manejan `FileNotFoundError` mostrando un `st.error()` si los archivos no están presentes.

### 3.3. Contenido de las Pestañas

#### Pestaña: Data Overview
* Muestra "Dataset Statistics" (`df.describe()`) y "Dataset Shape" (`df.shape`) en dos columnas.
* Incluye una distribución de las posiciones de los jugadores (`df['Position'].value_counts()`) como un gráfico de barras.
* Presenta una "Data Sample" (`df.head(10)`).
* Cada sección está envuelta en un `div` con la clase `card` para aplicar el estilo CSS.

#### Pestaña: Player Stats Analysis
* Organizada en filas de dos columnas.
* **Points Distribution:** Histograma (`px.histogram`) de `Total_Points`.
* **Points per Position:** Diagrama de caja (`px.box`) de `Total_Points` por `Position`.
* **Points vs Age:** Gráfico de dispersión (`px.scatter`) de `Age` vs `Total_Points`.
* **Correlation Heatmap:** Matriz de correlación (`px.imshow`) de las columnas numéricas de `model_data`.
* **Feature Explorer:**
    * Un `st.selectbox()` permite al usuario elegir una característica numérica de `model_data`.
    * Se muestra un gráfico de dispersión (`px.scatter`) de la característica seleccionada contra `Total_Points` (o la última variable numérica si `Total_Points` no existe).

#### Pestaña: Model Performance
* Similar a la pestaña de Dash, muestra:
    * **Error Metrics Comparison:** Gráfico de barras (`px.bar`) de `Test_MSE` y `Test_MAE`.
    * **R² Score by Model:** Gráfico de barras (`px.bar`) de `Test_R2`.
    * **Performance vs Training Time:** Gráfico de dispersión (`px.scatter`).
    * **Model Metrics Details:** `st.dataframe(results_df.round(3))`.

#### Pestaña: Predict Points
* Utiliza `st.columns(3)` para organizar los campos de entrada (`st.number_input`).
* **Manejo del Estado con `st.session_state`:** Los valores de los campos de entrada se inicializan y mantienen en `st.session_state` para que no se reseteen en cada re-ejecución del script (por ejemplo, después de hacer clic en el botón de predicción).
    * `if 'minutes' not in st.session_state: st.session_state.update({...})` inicializa los valores.
    * Los `st.number_input` usan `value=st.session_state.nombre_variable`.
* Un `st.button("Predict Points", type="primary")` activa la lógica de predicción.
* **Lógica de Predicción (dentro del `if predict_button ...`):**
    1.  Se muestra un `st.spinner('Calculating prediction...')`.
    2.  Se prepara un DataFrame (`input_data`) con los valores ingresados por el usuario.
    3.  Se asegura que `input_data` tenga todas las `model_features` requeridas, rellenando con 0 si es necesario y reordenando las columnas.
    4.  Se itera sobre `trained_models` para obtener predicciones individuales. Las predicciones negativas se ajustan a 0.
    5.  Si hay predicciones, se muestra la media (`avg_prediction`) con `st.success()`.
    6.  Se genera un gráfico de barras (`px.bar`) comparando las predicciones de cada modelo y se muestra con `st.plotly_chart()`.
    7.  Se manejan errores si la predicción falla.

## 4. Ejecución de la Aplicación

Para ejecutar este dashboard localmente:

1.  Asegúrate de tener todas las dependencias instaladas (principalmente `streamlit`, `pandas`, `plotly`, `joblib`).
2.  Coloca los archivos de datos (`2023_nba_player_stats.csv`, `database.csv`, `Model comparison results.csv`) y la carpeta `models` (con `top_three_models.pkl` y `model_features.pkl`) en el mismo directorio que `streamlit_app.py`, o ajusta las rutas en el script. Asegúrate también de que `RenameDatabase.py` esté accesible.
3.  Abre una terminal en el directorio del proyecto.
4.  Ejecuta el comando: `streamlit run streamlit_app.py`
5.  Streamlit usualmente abrirá la aplicación en tu navegador web predeterminado (la URL suele ser `http://localhost:8501`).

## 5. Consideraciones Adicionales

* **CSS Personalizado:** El uso de `st.markdown("""<style>...</style>""", unsafe_allow_html=True)` es una forma efectiva de aplicar estilos personalizados rápidamente en Streamlit.
* **Interactividad:** Streamlit maneja la reactividad de forma implícita. Cuando un widget cambia su valor, el script se re-ejecuta, y Streamlit actualiza la interfaz de manera eficiente.
* **Funciones de Caching:** El uso de `@st.cache_data` y `@st.cache_resource` es crucial para el rendimiento en aplicaciones Streamlit que manejan datos o modelos grandes.

## Código de la aplicación
```python
import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import warnings
warnings.filterwarnings('ignore') # Ignorar advertencias

# --- Configuración Inicial de la Página ---
# st.set_page_config() debe ser la primera comando de Streamlit en el script.
# page_title: Título que aparece en la pestaña del navegador.
# layout="wide": Hace que el contenido ocupe todo el ancho de la página.
st.set_page_config(page_title="NBA Player Stats Analysis", layout="wide")

# --- CSS Personalizado ---
# Se inyecta CSS para dar un estilo más personalizado a la aplicación.
# unsafe_allow_html=True permite incrustar HTML y CSS crudo.
st.markdown("""
    <style>
    .main-header {font-size:2.5rem; color:#1D428A; text-align:center; margin-bottom:1rem;}
    .sub-header {font-size:1.8rem; color:#1D428A; margin-top:1rem; margin-bottom:1rem;}
    .card {background-color:#f9f9f9; padding:1rem; border-radius:5px; box-shadow:0 2px 5px rgba(0,0,0,0.1); margin-bottom:1rem;}
    </style>
    """, unsafe_allow_html=True)

# --- Funciones de Carga de Datos y Modelos con Caching ---

# @st.cache_data es un decorador para cachear el resultado de funciones que retornan datos (ej. DataFrames).
# Esto evita recargar y reprocesar los datos en cada interacción del usuario, mejorando el rendimiento.
@st.cache_data
def load_data():
    """Carga y cachea todos los archivos de datos CSV requeridos."""
    try:
        # Cargar el conjunto de datos principal
        df_streamlit = pd.read_csv('2023_nba_player_stats.csv')
        # Importar la función de renombrado desde un archivo separado (asumiendo que existe)
        from RenameDatabase import renameDatabase
        df_streamlit = renameDatabase(df_streamlit)
        
        # Cargar datos adicionales para el modelo y resultados de comparación
        model_data_st = pd.read_csv('database.csv')
        results_df_st = pd.read_csv('Model comparison results.csv')
        
        return df_streamlit, model_data_st, results_df_st
    except Exception as e:
        st.error(f"Error al cargar los datos: {e}") # Mostrar mensaje de error en la app
        return None, None, None # Retornar None si hay error

# @st.cache_resource es un decorador para cachear recursos que no son fácilmente serializables (ej. modelos de ML, conexiones a BD).
@st.cache_resource
def load_models():
    """Carga y cachea los modelos pre-entrenados y sus características."""
    try:
        trained_models_st = joblib.load('models/top_three_models.pkl')
        model_features_st = joblib.load('models/model_features.pkl')
        return trained_models_st, model_features_st
    except Exception as e:
        st.error(f"Error al cargar los modelos: {e}")
        return None, None

# --- Título Principal y Carga de Archivos ---
# Mostrar el título principal de la aplicación usando la clase CSS personalizada.
st.markdown("<h1 class='main-header'>NBA Player Stats Analysis Dashboard</h1>", unsafe_allow_html=True)

# Llamar a las funciones para cargar datos y modelos.
df, model_data, results_df = load_data()
trained_models, model_features = load_models()

# Verificar si la carga de datos fue exitosa. Si no, detener la ejecución de la app.
if df is None or model_data is None or results_df is None:
    st.error("Fallo al cargar los archivos de datos requeridos. Por favor, verifique que todos los archivos CSV existan.")
    st.stop() # Detiene la ejecución del script de Streamlit

# Verificar si la carga de modelos fue exitosa.
if trained_models is None or model_features is None:
    st.error("Fallo al cargar los archivos de modelo. Por favor, verifique que los archivos de modelo existan en el directorio 'models'.")
    # No se detiene la app aquí, podría funcionar parcialmente sin modelos.

# --- Creación de Pestañas (Tabs) ---
# st.tabs() crea un contenedor de pestañas.
tab1, tab2, tab3, tab4 = st.tabs(["Data Overview", "Player Stats Analysis", "Model Performance", "Predict Points"])

# --- Contenido de la Pestaña 1: Data Overview ---
with tab1: # Contexto para la primera pestaña
    st.markdown("<h2 class='sub-header'>NBA Player Stats Overview</h2>", unsafe_allow_html=True)
    
    # st.columns() crea un layout de columnas.
    col1, col2 = st.columns(2) # Dos columnas de igual ancho
    
    with col1: # Contenido de la primera columna
        st.markdown("<div class='card'>", unsafe_allow_html=True) # Inicio de la tarjeta estilizada
        st.subheader("Dataset Statistics")
        st.dataframe(df.describe().round(2), use_container_width=True) # Muestra estadísticas descriptivas
        st.markdown("</div>", unsafe_allow_html=True) # Fin de la tarjeta
    
    with col2: # Contenido de la segunda columna
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Dataset Shape")
        st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}") # Muestra dimensiones del DataFrame
        
        if 'Position' in df.columns: # Verificar si la columna 'Position' existe
            st.subheader("Position Distribution")
            position_counts = df['Position'].value_counts().reset_index()
            position_counts.columns = ['Position', 'Count'] # Renombrar columnas para el gráfico
            st.bar_chart(position_counts, x='Position', y='Count') # Gráfico de barras de la distribución de posiciones
        st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Data Sample")
    st.dataframe(df.head(10), use_container_width=True) # Muestra las primeras 10 filas del DataFrame
    st.markdown("</div>", unsafe_allow_html=True)

# --- Contenido de la Pestaña 2: Player Stats Analysis ---
with tab2:
    st.markdown("<h2 class='sub-header'>Player Stats Analysis</h2>", unsafe_allow_html=True)
    
    # Primera fila de gráficos
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Points Distribution")
        if 'Total_Points' in df.columns:
            # Histograma de la distribución de puntos totales
            fig = px.histogram(df, x='Total_Points', 
                              title='Distribution of Total Points',
                              color_discrete_sequence=['#C8102E']) # Color de las barras
            st.plotly_chart(fig, use_container_width=True) # Mostrar gráfico Plotly
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        if 'Position' in df.columns and 'Total_Points' in df.columns:
            st.subheader("Points per Position")
            # Diagrama de caja de puntos por posición
            fig = px.box(df, x='Position', y='Total_Points',
                        title='Points Distribution by Position',
                        color='Position') # Colorear por posición
            st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Segunda fila de gráficos
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        if 'Age' in df.columns and 'Total_Points' in df.columns:
            st.subheader("Points vs Age")
            color_col = 'Position' if 'Position' in df.columns else None # Usar 'Position' para color si existe
            # Gráfico de dispersión de Edad vs Puntos, con línea de tendencia
            fig = px.scatter(df, x='Age', y='Total_Points',
                            title='Age vs Total Points',
                            color=color_col,
                            trendline='ols') # 'ols' para Ordinary Least Squares
            st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Correlation Heatmap")
        # Mapa de calor de la matriz de correlación (usando model_data)
        numeric_data = model_data.select_dtypes(include=['number']) # Seleccionar solo columnas numéricas
        fig = px.imshow(numeric_data.corr(),
                       title='Feature Correlation Matrix',
                       color_continuous_scale='RdBu_r') # Esquema de color
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Explorador de Características
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Feature Explorer")
    
    numeric_cols = model_data.select_dtypes(include=['number']).columns.tolist()
    target_var = 'Total_Points' if 'Total_Points' in model_data.columns else numeric_cols[-1] if numeric_cols else None
    feature_options = [col for col in numeric_cols if col != target_var] if numeric_cols else []
    
    if feature_options: # Si hay opciones de características para seleccionar
        default_feature = 'Field_Goals_Attempted' if 'Field_Goals_Attempted' in feature_options else feature_options[0]
        default_idx = feature_options.index(default_feature) if default_feature in feature_options else 0
        
        # Menú desplegable para seleccionar una característica
        selected_feature = st.selectbox(
            f"Select a feature to analyze its relationship with {target_var}:", 
            options=feature_options,
            index=default_idx
        )
        
        color_var = 'Position' if 'Position' in model_data.columns else None
        
        # Gráfico de dispersión de la característica seleccionada vs variable objetivo
        fig = px.scatter(model_data, x=selected_feature, y=target_var,
                        color=color_var,
                        title=f'{selected_feature} vs {target_var}',
                        trendline='ols')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.write("No numeric features available for exploration in model_data.")
    st.markdown("</div>", unsafe_allow_html=True)

# --- Contenido de la Pestaña 3: Model Performance ---
with tab3:
    st.markdown("<h2 class='sub-header'>Model Performance Analysis</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Error Metrics Comparison")
        # Gráfico de barras comparando Test_MSE y Test_MAE por modelo
        fig = px.bar(results_df, x='Model', y=['Test_MSE', 'Test_MAE'],
                     barmode='group', # Barras agrupadas
                     title='Error Metrics by Model',
                     color_discrete_sequence=['#C8102E', '#1D428A'])
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("R² Score by Model")
        # Gráfico de barras de la puntuación R² por modelo
        fig = px.bar(results_df, x='Model', y='Test_R2',
                     title='R² Score by Model',
                     color='Test_R2',
                     color_continuous_scale='viridis')
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Gráfico de Rendimiento vs Tiempo de Entrenamiento
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Performance vs Training Time")
    fig = px.scatter(results_df, x='Training_Time', y='Test_MAE',
                     size='Test_MSE', size_max=50, # Tamaño de los puntos basado en Test_MSE
                     hover_name='Model', # Información al pasar el mouse
                     title='Model Performance vs Training Time',
                     labels={'Training_Time': 'Training Time (s)', 'Test_MAE': 'Absolute Error'}, # Etiquetas de ejes
                     color='Model')
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Tabla con detalles de las métricas del modelo
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Model Metrics Details")
    st.dataframe(results_df.round(3), use_container_width=True) # Mostrar DataFrame con resultados redondeados
    st.markdown("</div>", unsafe_allow_html=True)

# --- Contenido de la Pestaña 4: Predict Points ---
with tab4:
    st.markdown("<h2 class='sub-header'>Player Points Predictor</h2>", unsafe_allow_html=True)
    
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Input Player Stats")
    
    # Usar st.session_state para mantener los valores de los inputs entre re-ejecuciones del script.
    # Esto es útil porque Streamlit re-ejecuta el script en cada interacción.
    if 'minutes' not in st.session_state: # Inicializar si no existen en el estado de sesión
        st.session_state.update({
            'minutes': 0, 'fga': 0, 'tpa': 0, 'fta': 0, 
            'ast': 0, 'stl': 0, 'tov': 0, 'reb': 0, 'pf': 0
        })
    
    # Columnas para organizar los campos de entrada
    col1, col2, col3 = st.columns(3)
    with col1:
        # st.number_input crea un campo para entrada numérica.
        # El valor se toma y actualiza desde st.session_state.
        minutes = st.number_input("Minutes Played:", min_value=0, value=st.session_state.minutes, key='minutes_input_key')
        fga = st.number_input("Field Goals Attempted:", min_value=0, value=st.session_state.fga, key='fga_input_key')
        tpa = st.number_input("Three Point Attempts:", min_value=0, value=st.session_state.tpa, key='tpa_input_key')
    
    with col2:
        fta = st.number_input("Free Throws Attempted:", min_value=0, value=st.session_state.fta, key='fta_input_key')
        ast = st.number_input("Assists:", min_value=0, value=st.session_state.ast, key='ast_input_key')
        stl = st.number_input("Steals:", min_value=0, value=st.session_state.stl, key='stl_input_key')
    
    with col3:
        tov = st.number_input("Turnovers:", min_value=0, value=st.session_state.tov, key='tov_input_key')
        reb = st.number_input("Total Rebounds:", min_value=0, value=st.session_state.reb, key='reb_input_key')
        pf = st.number_input("Personal Fouls:", min_value=0, value=st.session_state.pf, key='pf_input_key')
    
    # Actualizar el estado de sesión con los valores actuales de los inputs
    # Esto es importante si los inputs se modifican y queremos que persistan para la próxima acción.
    st.session_state.minutes = minutes
    st.session_state.fga = fga
    st.session_state.tpa = tpa
    st.session_state.fta = fta
    st.session_state.ast = ast
    st.session_state.stl = stl
    st.session_state.tov = tov
    st.session_state.reb = reb
    st.session_state.pf = pf
    
    # Botón para activar la predicción
    predict_button = st.button("Predict Points", type="primary") # type="primary" para un estilo destacado
    
    # Lógica que se ejecuta si el botón es presionado y los modelos están cargados
    if predict_button and trained_models and model_features:
        with st.spinner('Calculating prediction...'): # Muestra un indicador de carga
            # Preparar los datos de entrada para el modelo
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
            input_df = pd.DataFrame(input_data_dict)
            
            # Asegurar que el DataFrame de entrada tenga todas las características requeridas por el modelo,
            # en el orden correcto, y rellenar con 0 las que falten.
            for col in model_features:
                if col not in input_df.columns:
                    input_df[col] = 0
            input_df = input_df[model_features] # Reordenar columnas según model_features
            
            # Realizar predicciones
            predictions = {}
            for name, model in trained_models.items():
                try:
                    pred = model.predict(input_df)[0]
                    predictions[name] = max(0, pred)  # Asegurar que la predicción no sea negativa
                except Exception as e:
                    st.error(f"Fallo al predecir con el modelo {name}: {e}")
            
            # Mostrar resultados
            if predictions:
                avg_prediction = round(sum(predictions.values()) / len(predictions), 1)
                st.success(f"**Puntos Predichos (promedio): {avg_prediction}**") # Mensaje de éxito
                
                # Crear DataFrame para el gráfico de comparación de predicciones
                pred_df = pd.DataFrame({
                    'Model': list(predictions.keys()),
                    'Predicted_Points': list(predictions.values())
                })
                
                # Gráfico de barras comparando las predicciones de cada modelo
                fig_pred_comp = px.bar(pred_df, x='Model', y='Predicted_Points', 
                                   title="Predicted Points by Different Models",
                                   color='Model')
                st.plotly_chart(fig_pred_comp, use_container_width=True)
            else:
                st.error("Fallo al realizar las predicciones.") # Mensaje de error

    st.markdown("</div>", unsafe_allow_html=True) # Fin de la tarjeta de predicción
    
# --- Pie de Página ---
st.markdown("""
<div style="text-align: center; margin-top: 50px; padding: 20px; background-color: #f0f2f6; border-radius: 5px;">
    <p>NBA Player Stats Analysis Dashboard - Created with Streamlit</p>
</div>
""", unsafe_allow_html=True)
```
