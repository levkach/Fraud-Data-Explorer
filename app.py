# -*- coding: utf-8 -*-
import pandas as pd
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import os

pd.set_option('max_rows', 50)
pd.set_option('max_columns', 10)
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']


# Finds csv files
def find_csv_filenames(path_to_dir, suffix=".csv"):
    filenames = os.listdir(path_to_dir)
    return [filename for filename in filenames if filename.endswith(suffix)]


# Takes raw data, produces frauds, nonfrauds based on given threshold or actually from the data.
def get_classificator_info(data, threshold=0.5, count_conf_actual=True):
    # tn = len(data[(data.y_true == 0.0) & (data.y_est == 0.0)].index)
    # fn = len(data[(data.y_true == 1.0) & (data.y_est == 0.0)].index)
    # tp = len(data[(data.y_true == 1.0) & (data.y_est if count_conf_actual else data.y_probability >= threshold)].index)
    # fp = len(data[(data.y_true == 0.0) & (data.y_est if count_conf_actual else data.y_probability >= threshold)].index)
    # prec = (1.0 * tp) / (tp + fp)
    # recall = (1.0 * tp) / (tp + fn)

    # Actual results
    actual_frauds = data[data.y_true == 1.0]
    actual_non_frauds = data[data.y_true == 0.0]
    # Predicted results
    predicted_frauds = data[data.y_probability >= threshold]
    predicted_non_frauds = data[data.y_probability < threshold]

    ret = {
        # 'tn': tn, 'fn': fn, 'tp': tp, 'fp': fp, 'precision': prec, 'recall': recall,
        'act_frauds': actual_frauds, 'act_non_frauds': actual_non_frauds,
        'pred_frauds': predicted_frauds, 'pred_non_frauds': predicted_non_frauds}
    return ret


def get_markers(data, threshold=0.5, with_non_frauds=True, actual_results=False):
    """
    Return scatters based on parameters

    :param data:
    :param threshold:
    :param with_non_frauds: include non-frauds
    :param actual_results: decide fraud/non-fraud, basing on the dataset's actual data (i.e. disregard threshold)
    :return:
    """
    # Train set. TODO: agree on standard format
    if len(raw_data.columns) != 8:
        print("Not standard data set.")
        return

    info = get_classificator_info(data, threshold)
    frauds = info["act_frauds"] if actual_results else info['pred_frauds']
    trace_frauds = go.Scatter3d(
        x=frauds["C1"],
        y=frauds["C2"],
        z=frauds["C3"],
        text=frauds["Fraud_amount"],
        hoverinfo='text',
        mode='markers',
        marker=dict(
            size=abs(3 + frauds["Fraud_amount"] / 5000),
            color='red',
            # colorscale='Hot',
            line=dict(
                color='rgba(100, 100, 217, 0.14)',
                width=0.5
            ),
            opacity=0.8
        )
    )
    nonfrauds = info["act_non_frauds"] if actual_results else info['pred_non_frauds']
    trace_nonfrauds = go.Scatter3d(
        x=nonfrauds["C1"],
        y=nonfrauds["C2"],
        z=nonfrauds["C3"],
        text=nonfrauds["y_probability"],
        hoverinfo='text',
        mode='markers',
        marker=dict(
            size=2,
            color='grey',
            line=dict(
                color='rgba(100, 100, 217, 0.14)',
                width=0.5
            ),
            opacity=0.8
        )
    )
    return [trace_frauds, trace_nonfrauds] if with_non_frauds else [trace_frauds]


graph_layout = go.Layout(
    margin=dict(
        l=0,
        r=0,
        b=0,
        t=0
    )
)

avaliable_data_sets = find_csv_filenames(os.path.join(os.path.curdir, 'data'))
print(avaliable_data_sets)

app = dash.Dash("Test", external_stylesheets=external_stylesheets)
raw_data = pd.read_csv("data/PCA_test_set.csv")

# Initial graph params


# Main layout
app.layout = html.Div([

    html.H2('Fraud data visualize'),

    # Choosing dataset
    html.Label('Dataset'),
    dcc.Dropdown(
        id='dataset',
        options=[{'label': i, 'value': i} for i in avaliable_data_sets],
        value=avaliable_data_sets[0]
    ),

    # With or without nonfrauds
    html.Label('Nonfrauds'),
    dcc.Checklist(
        id='nonfrauds',
        options=[
            {'label': 'With Nonfrauds', 'value': 'Yes'},
        ],
        values=[]
    ),

    # Highlights actual frauds
    html.Label('Highlight actual frauds'),
    dcc.Checklist(
        id='highlight',
        options=[
            {'label': 'Highlight', 'value': 'Yes'},
        ],
        values=[]
    ),

    # Threshold slider
    html.Label('Threshold'),
    dcc.Slider(
        id='threshold',
        min=0.1,
        max=1.0,
        step=0.05,
        value=0.5,
    ),
    html.Div(id="threshold-selected"),

    # Update button
    html.Button("Reload", id='reload'),

    # Graph
    html.Label("Graph"),
    html.Div([
        dcc.Graph(
            id='graph',
            figure=go.Figure(
                data=get_markers(raw_data, actual_results=True),
                layout=graph_layout,
            ),
        ),
    ], id="graph-container"),

], style={'width': '80%'})


@app.callback(Output('threshold-selected', 'children'),
              [Input('threshold', 'value')])
def update_slider(value):
    return "Selected threshold: {}".format(value)


@app.callback(Output('graph-container', 'children'),
              [Input('reload', 'n_clicks')],
              [State('dataset', 'value'),
               State('threshold', 'value'),
               State('nonfrauds', 'values'),
               State('highlight', 'values')]
              )
def update_graph(n_clicks, dataset, threshold, with_non_frauds, highlight):
    print(n_clicks)
    new_data = pd.read_csv("data/" + dataset)
    data = get_markers(new_data, threshold=threshold,
                       with_non_frauds=len(with_non_frauds) != 0,
                       actual_results=len(highlight) != 0)
    return html.Div([
        dcc.Graph(
            id='graph',
            figure=go.Figure(
                data=data,
                layout=graph_layout
            )
        )
    ], id="graph-container")


app.run_server(debug=True)
