import dash
from dash import dcc, html, Input, Output, dash_table
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import joblib
from RenameDatabase import renameDatabase

import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('2023_nba_player_stats.csv')
df = renameDatabase(df)
model_data = pd.read_csv('database.csv')
results_df = pd.read_csv('Model comparison results.csv')

try:
    trained_models = joblib.load('models/top_three_models.pkl')
    model_features = joblib.load('models/model_features.pkl')
    print("Modelos cargados correctamente:")
    for name in trained_models.keys():
        print(f"- {name}")
except Exception as e:
    print(f"Error al cargar los modelos: {e}")
    trained_models = {}
    model_features = []

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.layout = dbc.Container([
    html.H1("NBA Player Stats Analysis Dashboard",
            style={'textAlign': 'center', 'color': '#1D428A', 'marginBottom':30, 'marginTop':20}),
    dcc.Tabs([
        dcc.Tab(label='Data Overview', children=[
            dbc.Row([
                dbc.Col([
                    html.H3("NBA Player Stats Overview", style={'margin': '20px 0px'}),
                    html.Div([
                        html.H4("Dataset Statistics"),
                        dash_table.DataTable(
                            data = df.describe().reset_index().round(2).to_dict('records'),
                            columns = [{"name": i, "id": i} for i in df.describe().reset_index().columns],
                            style_table = {'overflowX': 'auto'},
                            style_cell = {'textAlign': 'center', 'padding': '10px'},
                            style_header = {'backgroundColor': '#1D428A', 'color': 'white', 'fontWeight': 'bold'}
                        )
                    ], style={'width': '100%', 'margin': '10px'}),
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

        dcc.Tab(label='Player Stats Analysis', children=[
            dbc.Row([
                dbc.Col([
                    html.H3("Player Stats Analysis", style={'margin': '20px 0px'}),
                    dbc.Row([
                        dbc.Col([
                            html.H4("Points Distribution"),
                            dcc.Graph(figure = px.histogram(df, x = 'Total_Points',
                                                            title = 'Distibution of Total Points',
                                                            color_discrete_sequence = ['#C8102E']))
                        ], width=6),
                        dbc.Col([
                            html.H4("Points per Position"),
                            dcc.Graph(figure = px.box(df, x ='Position', y = 'Total_Points',
                                                      title = 'Points Distribution by Position',
                                                      color = 'Position'))
                        ], width=6),
                    ]),
                    dbc.Row([
                        dbc.Col([
                            html.H4("Points vs Age"),
                            dcc.Graph(figure = px.scatter(df, x = 'Age', y = 'Total_Points',
                                                          title = 'Age vs Total Points',
                                                          color = 'Position',
                                                          trendline = 'ols'))
                        ], width=6),
                        dbc.Col([
                            html.H4("Correlation Heatmap"),
                            dcc.Graph(figure = px.imshow(model_data.corr(),
                                                         title = 'Feature Correlation Matrix',
                                                         color_continuous_scale = 'RdBu_r'))
                        ], width=6),
                    ]),
                    dbc.Row([
                        dbc.Col([
                            html.H4("Feature Explorer"),
                            html.P("Select a feature to analyze its relationship with Total Points:"),
                            dcc.Dropdown(
                                id = 'feature-dropdown',
                                options = [{'label': col, 'value': col} for col in model_data.columns if col != 'Total_Points'],
                                value = 'Field_Goals_Attempted',
                                style = {'width': '50%'}),
                            dcc.Graph(id='feature-points-scatter')
                        ])
                    ])
                ])
            ])
        ]),

        dcc.Tab(label='Model Performance', children=[
            dbc.Row([
                dbc.Col([
                    html.H3("Model Performance Analysis", style={'margin': '20px 0px'}),
                    dbc.Row([
                        dbc.Col([
                            html.H4("Error Metrics Comparison"),
                            dcc.Graph(figure = px.bar(results_df, x = 'Model', y = ['Test_MSE', 'Test_MAE'],
                                                      barmode = 'group',
                                                      title = 'Error Metrics by Model',
                                                      color_discrete_sequence = ['#C8102E', '#1D428A']))
                        ], width=6),
                        dbc.Col([
                            html.H4("R² Score by Model"),
                            dcc.Graph(figure = px.bar(results_df, x = 'Model', y = 'Test_R2',
                                                      title = 'R² Score by Model',
                                                      color = 'Test_R2',
                                                      color_continuous_scale = 'viridis'))
                        ], width=6),
                    ]),
                    dbc.Row([
                        dbc.Col([
                            html.H4("Performance vs Training Time"),
                            dcc.Graph(figure = px.scatter(results_df, x = 'Training_Time', y = 'Test_MAE',
                                                          size = 'Test_MSE', size_max = 50,
                                                          hover_name = 'Model',
                                                          title = 'Model Performance vs Training Time',
                                                          labels = {'Training_Time': 'Training Time (s)', 'Test_MAE': 'Absolute Error'},
                                                          color = 'Model'))
                        ])
                    ]),
                    dbc.Row([
                        dbc.Col([
                            html.H4("Model Metrics Details"),
                            dash_table.DataTable(
                                data = results_df.round(3).to_dict('records'),
                                columns = [{"name": i, "id": i} for i in results_df.columns],
                                style_table = {'overflowX': 'auto'},
                                style_cell = {'textAlign': 'center', 'padding': '10px'},
                                style_header = {'backgroundColor': '#1D428A', 'color': 'white', 'fontWeight': 'bold'},
                                sort_action = 'native',
                                filter_action = 'native'
                            )
                        ])
                    ])
                ])
            ])
        ]),

        dcc.Tab(label = 'Predict Points', children=[
            dbc.Row([
                dbc.Col([
                    html.H3("Player Points Predictor", style={'margin': '20px 0px'}),
                    dbc.Card([
                        dbc.CardHeader(html.H4("Input Player Stats")),
                        dbc.CardBody([
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
                            ], className = "mb-3"),
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
                            dbc.Button('Predict Points', id = 'predict-button',
                                       color = 'danger', className = 'mt-3'),
                            html.Div(id = 'prediction-output',
                                     style = {'margin': '20px 0', 'fontWeight': 'bold', 'fontSize': '18px'})
                        ])
                    ], style = {'marginBottom': '20px'}),
                    html.Div([
                        html.H4("Prediction Comparison"),
                        dcc.Graph(id='prediction-comparison')
                    ])
                ])
            ])
        ])
    ], style={'marginTop': '20px'})
], fluid=True)

@app.callback(
    Output('feature-points-scatter', 'figure'),
    [Input('feature-dropdown', 'value')]
)
def update_features_scatter(feature):
    fig = px.scatter(df, x = feature, y = 'Total_Points',
                     color = 'Position',
                     title = f'{feature} vs Total Points',
                     trendline = 'ols')
    return fig

@app.callback(
    [Output('prediction-output', 'children'),
     Output('prediction-comparison', 'figure')],
    [Input('predict-button', 'n_clicks')],
    [dash.dependencies.State('minutes-input', 'value'),
     dash.dependencies.State('fga-input', 'value'),
     dash.dependencies.State('3pa-input', 'value'),
     dash.dependencies.State('fta-input', 'value'),
     dash.dependencies.State('ast-input', 'value'),
     dash.dependencies.State('stl-input', 'value'),
     dash.dependencies.State('tov-input', 'value'),
     dash.dependencies.State('reb-input', 'value'),
     dash.dependencies.State('pf-input', 'value')]
)
def predict_points(n_clicks, minutes, fga, tpa, fta, ast, stl, tov, reb, pf):
    if n_clicks is None or n_clicks == 0:
        return "Waiting data input...", go.Figure()

    if not trained_models or not model_features:
        return "Error to load data", go.Figure()

    input_values = [minutes, fga, tpa, fta, ast, stl, tov, reb, pf]
    if any(x is None for x in input_values):
        return "Please enter all the fields.", go.Figure()

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

    for col in model_features:
        if col not in input_data.columns:
            input_data[col] = 0
    input_data = input_data.reindex(columns=model_features, fill_value=0)

    predictions = {}
    for name, model in trained_models.items():
        try:
            pred = model.predict(input_data)[0]
            predictions[name] = max(0, pred)
        except Exception as e:
            print(f"Failed to prediction with model {name}: {e}")
            predictions[name] = 0

    fig = go.Figure()
    for model_name, pred in predictions.items():
        fig.add_trace(go.Bar(
            x = [model_name],
            y = [pred],
            name = model_name,
        ))
    fig.update_layout(
        title = "Predicted Points by Different models",
        yaxis_title = 'Predicted Points',
        barmode = 'group'

    )
    if predictions:
        avg_prediction = round(sum(predictions.values()) / len(predictions), 1)
        result_text = f"Predicted Points (mean): {avg_prediction}"
    else:
        result_text = "Failed to make the predictions"
    
    return result_text, fig

if __name__ == '__main__':
    app.run(debug=True, port=8050)
