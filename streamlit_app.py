import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import warnings
warnings.filterwarnings('ignore')

# Set page configuration once at the beginning
st.set_page_config(page_title="NBA Player Stats Analysis", layout="wide")

# Custom CSS to style the app with NBA colors
st.markdown("""
    <style>
    .main-header {font-size:2.5rem; color:#1D428A; text-align:center; margin-bottom:1rem;}
    .sub-header {font-size:1.8rem; color:#1D428A; margin-top:1rem; margin-bottom:1rem;}
    .card {background-color:#f9f9f9; padding:1rem; border-radius:5px; box-shadow:0 2px 5px rgba(0,0,0,0.1); margin-bottom:1rem;}
    </style>
    """, unsafe_allow_html=True)

# Helper function for loading and processing data - externalized
@st.cache_data
def load_data():
    """Load and cache all required data files"""
    try:
        # Load the main dataset
        df = pd.read_csv('2023_nba_player_stats.csv')
        # Import the rename function from separate file
        from RenameDatabase import renameDatabase
        df = renameDatabase(df)
        
        # Load model data and results
        model_data = pd.read_csv('database.csv')
        results_df = pd.read_csv('Model comparison results.csv')
        
        return df, model_data, results_df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None

# Load models with caching
@st.cache_resource
def load_models():
    """Load and cache the pre-trained models"""
    try:
        trained_models = joblib.load('models/top_three_models.pkl')
        model_features = joblib.load('models/model_features.pkl')
        return trained_models, model_features
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None

# App title 
st.markdown("<h1 class='main-header'>NBA Player Stats Analysis Dashboard</h1>", unsafe_allow_html=True)

# Load data and models - assuming files exist as mentioned
df, model_data, results_df = load_data()
trained_models, model_features = load_models()

# Check if data loading was successful
if df is None or model_data is None or results_df is None:
    st.error("Failed to load required data files. Please check that all CSV files exist.")
    st.stop()

if trained_models is None or model_features is None:
    st.error("Failed to load model files. Please check that model files exist in the models directory.")

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs(["Data Overview", "Player Stats Analysis", "Model Performance", "Predict Points"])

# Tab 1: Data Overview
with tab1:
    st.markdown("<h2 class='sub-header'>NBA Player Stats Overview</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Dataset Statistics")
        st.dataframe(df.describe().round(2), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Dataset Shape")
        st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
        
        if 'Position' in df.columns:
            st.subheader("Position Distribution")
            position_counts = df['Position'].value_counts().reset_index()
            position_counts.columns = ['Position', 'Count']
            st.bar_chart(position_counts, x='Position', y='Count')
        st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Data Sample")
    st.dataframe(df.head(10), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# Tab 2: Player Stats Analysis
with tab2:
    st.markdown("<h2 class='sub-header'>Player Stats Analysis</h2>", unsafe_allow_html=True)
    
    # First row
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Points Distribution")
        if 'Total_Points' in df.columns:
            fig = px.histogram(df, x='Total_Points', 
                              title='Distribution of Total Points',
                              color_discrete_sequence=['#C8102E'])
            st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        if 'Position' in df.columns and 'Total_Points' in df.columns:
            st.subheader("Points per Position")
            fig = px.box(df, x='Position', y='Total_Points',
                        title='Points Distribution by Position',
                        color='Position')
            st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Second row
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        if 'Age' in df.columns and 'Total_Points' in df.columns:
            st.subheader("Points vs Age")
            color_col = 'Position' if 'Position' in df.columns else None
            fig = px.scatter(df, x='Age', y='Total_Points',
                            title='Age vs Total Points',
                            color=color_col,
                            trendline='ols')
            st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Correlation Heatmap")
        # Use model_data for correlation analysis
        numeric_data = model_data.select_dtypes(include=['number'])
        fig = px.imshow(numeric_data.corr(),
                       title='Feature Correlation Matrix',
                       color_continuous_scale='RdBu_r')
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Feature explorer
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Feature Explorer")
    
    numeric_cols = model_data.select_dtypes(include=['number']).columns.tolist()
    target_var = 'Total_Points' if 'Total_Points' in model_data.columns else numeric_cols[-1]
    feature_options = [col for col in numeric_cols if col != target_var]
    
    if feature_options:
        default_feature = 'Field_Goals_Attempted' if 'Field_Goals_Attempted' in feature_options else feature_options[0]
        default_idx = feature_options.index(default_feature)
        
        selected_feature = st.selectbox(
            f"Select a feature to analyze its relationship with {target_var}:", 
            options=feature_options,
            index=default_idx)
        
        color_var = 'Position' if 'Position' in model_data.columns else None
        
        fig = px.scatter(model_data, x=selected_feature, y=target_var,
                        color=color_var,
                        title=f'{selected_feature} vs {target_var}',
                        trendline='ols')
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

# Tab 3: Model Performance
with tab3:
    st.markdown("<h2 class='sub-header'>Model Performance Analysis</h2>", unsafe_allow_html=True)
    
    # First row
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Error Metrics Comparison")
        fig = px.bar(results_df, x='Model', y=['Test_MSE', 'Test_MAE'],
                     barmode='group',
                     title='Error Metrics by Model',
                     color_discrete_sequence=['#C8102E', '#1D428A'])
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("R² Score by Model")
        fig = px.bar(results_df, x='Model', y='Test_R2',
                     title='R² Score by Model',
                     color='Test_R2',
                     color_continuous_scale='viridis')
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Performance vs Training Time
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Performance vs Training Time")
    fig = px.scatter(results_df, x='Training_Time', y='Test_MAE',
                     size='Test_MSE', size_max=50,
                     hover_name='Model',
                     title='Model Performance vs Training Time',
                     labels={'Training_Time': 'Training Time (s)', 'Test_MAE': 'Absolute Error'},
                     color='Model')
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Model details table
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Model Metrics Details")
    st.dataframe(results_df.round(3), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# Tab 4: Predict Points
with tab4:
    st.markdown("<h2 class='sub-header'>Player Points Predictor</h2>", unsafe_allow_html=True)
    
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Input Player Stats")
    
    # Use session state to maintain form values between reruns
    if 'minutes' not in st.session_state:
        st.session_state.update({
            'minutes': 0, 'fga': 0, 'tpa': 0, 'fta': 0, 
            'ast': 0, 'stl': 0, 'tov': 0, 'reb': 0, 'pf': 0
        })
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        minutes = st.number_input("Minutes Played:", min_value=0, value=st.session_state.minutes)
        fga = st.number_input("Field Goals Attempted:", min_value=0, value=st.session_state.fga)
        tpa = st.number_input("Three Point Attempts:", min_value=0, value=st.session_state.tpa)
    
    with col2:
        fta = st.number_input("Free Throws Attempted:", min_value=0, value=st.session_state.fta)
        ast = st.number_input("Assists:", min_value=0, value=st.session_state.ast)
        stl = st.number_input("Steals:", min_value=0, value=st.session_state.stl)
    
    with col3:
        tov = st.number_input("Turnovers:", min_value=0, value=st.session_state.tov)
        reb = st.number_input("Total Rebounds:", min_value=0, value=st.session_state.reb)
        pf = st.number_input("Personal Fouls:", min_value=0, value=st.session_state.pf)
    
    # Store current values in session state
    current_values = {'minutes': minutes, 'fga': fga, 'tpa': tpa, 'fta': fta, 
                      'ast': ast, 'stl': stl, 'tov': tov, 'reb': reb, 'pf': pf}
    
    predict_button = st.button("Predict Points", type="primary")
    
    if predict_button and trained_models and model_features:
        with st.spinner('Calculating prediction...'):
            # Prepare input data
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
            input_data = pd.DataFrame(input_data_dict)
            
            # Ensure all required features are present
            for col in model_features:
                if col not in input_data.columns:
                    input_data[col] = 0
            input_data = input_data[model_features]  # Reindex with only needed columns
            
            # Make predictions
            predictions = {}
            for name, model in trained_models.items():
                try:
                    pred = model.predict(input_data)[0]
                    predictions[name] = max(0, pred)  # No negative points
                except Exception as e:
                    st.error(f"Failed to predict with model {name}: {e}")
            
            # Display results
            if predictions:
                avg_prediction = round(sum(predictions.values()) / len(predictions), 1)
                st.success(f"**Predicted Points (mean): {avg_prediction}**")
                
                # Create prediction comparison chart
                pred_df = pd.DataFrame({
                    'Model': list(predictions.keys()),
                    'Predicted_Points': list(predictions.values())
                })
                
                fig = px.bar(pred_df, x='Model', y='Predicted_Points', 
                           title="Predicted Points by Different Models",
                           color='Model')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("Failed to make predictions")

    st.markdown("</div>", unsafe_allow_html=True)
    
# Add a footer
st.markdown("""
<div style="text-align: center; margin-top: 50px; padding: 20px; background-color: #f0f2f6; border-radius: 5px;">
    <p>NBA Player Stats Analysis Dashboard - Created with Streamlit</p>
</div>
""", unsafe_allow_html=True)
