# Documentación: Dashboard de Análisis de Estadísticas de Jugadores de la NBA (Streamlit)

Este documento detalla la estructura y funcionamiento de la aplicación Streamlit desarrollada para analizar y predecir estadísticas de jugadores de la NBA, basada en el archivo `streamlit_app.py`.

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

