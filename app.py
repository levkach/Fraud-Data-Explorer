# -*- coding: utf-8 -*-
import json

import pandas as pd
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import os
import ipdb

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']


# Finds csv files
def find_csv_filenames(path_to_dir, suffix=".csv"):
    filenames = os.listdir(path_to_dir)
    return [filename for filename in filenames if filename.endswith(suffix)]


# Takes raw data, produces frauds, nonfrauds based on given threshold or actually from the data.
def get_classificator_info(data, threshold=0.5):
    # Train set. TODO: agree on dataset standard format
    if len(raw_data.columns) != 8:
        print("Not standard data set.")
        return

    # Actual results
    actual_frauds = data[data.y_true == 1.0]
    actual_non_frauds = data[data.y_true == 0.0]

    # Predicted results
    predicted_frauds = data[data.y_probability >= threshold]
    predicted_non_frauds = data[data.y_probability < threshold]

    ret = {'act_frauds': actual_frauds, 'act_non_frauds': actual_non_frauds,
           'pred_frauds': predicted_frauds, 'pred_non_frauds': predicted_non_frauds}
    return ret


def get_markers(data, threshold=0.5, with_non_frauds=True):
    """
    Return scatters based on parameters

    :param data:
    :param threshold:
    :param with_non_frauds: include non-frauds
    :param actual_results: decide fraud/non-fraud, basing on the dataset's actual data (i.e. disregard threshold)
    :return:
    """

    info = get_classificator_info(data, threshold)
    frauds = info['pred_frauds']
    trace_frauds = go.Scatter3d(
        x=frauds["C1"],
        y=frauds["C2"],
        z=frauds["C3"],
        name='Frauds',
        customdata=frauds.to_dict('records'),
        mode='markers',
        marker=dict(
            size=abs(3 + frauds["Fraud_amount"] / 6000),
            color='red',
            # colorscale='Hot',
            line=dict(
                color='rgba(100, 100, 217, 0.14)',
                width=0.5
            ),
            opacity=0.8
        )
    )
    nonfrauds = info['pred_non_frauds']
    trace_nonfrauds = go.Scatter3d(
        x=nonfrauds["C1"],
        y=nonfrauds["C2"],
        z=nonfrauds["C3"],
        name='Non-frauds',
        mode='markers',
        customdata=nonfrauds.to_dict('records'),
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

app = dash.Dash("Semestralka", external_stylesheets=external_stylesheets)
raw_data = pd.read_csv("data/PCA_test_set.csv")

# Initial graph params


# Main layout
app.layout = html.Div([

    html.H2('Fraud data visualize'),

    # Graph
    html.Label("Graph"),
    html.Div([
        dcc.Graph(
            id='graph',
            figure=go.Figure(
                data=get_markers(raw_data),
                layout=graph_layout,
            ),
        ),
        # Marker information
        html.Div(
            id='marker-info',
        )
    ], id="graph-container"),

    # Threshold slider
    html.Div(id="threshold-selected"),

    html.Label('Threshold'),
    html.Div([
        dcc.Slider(
            id='threshold',
            min=0.1,
            max=1.0,
            step=0.05,
            value=0.5,
        ),
    ], style={'width': '80%'}),
    html.Div([
        # With or without nonfrauds
        html.Label('Non-frauds'),
        dcc.Checklist(
            id='nonfrauds',
            options=[
                {'label': 'With Nonfrauds', 'value': 'Yes'},
            ],
            values=['Yes']
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

        # Choosing dataset
        html.Label('Dataset'),
        dcc.Dropdown(
            id='dataset',
            options=[{'label': i, 'value': i} for i in avaliable_data_sets],
            value=avaliable_data_sets[0],
            style={'width': '50%'}
        ),
    ]),
])


@app.callback(Output('threshold-selected', 'children'),
              [Input('threshold', 'value')])
def update_slider(value):
    return "Selected threshold: {}".format(value)


@app.callback(Output('graph', 'figure'),
              [Input('dataset', 'value'),
               Input('threshold', 'value'),
               Input('nonfrauds', 'values'),
               Input('highlight', 'values')]
              )
def update_graph(dataset, threshold, with_non_frauds, highlight):
    new_data = pd.read_csv("data/" + dataset)
    data = get_markers(new_data, threshold=threshold,
                       with_non_frauds=len(with_non_frauds) != 0)
    return go.Figure(
        data=data,
        layout=graph_layout
    )


@app.callback(Output('marker-info', 'children'),
              [Input('graph', 'clickData')])
def update_marker_info(selected_data):
    if selected_data is None:
        return ''
    # Get marker's custom data and clean it up
    marker_info = cleanup_marker_data(selected_data)
    return json.dumps(marker_info)


def cleanup_marker_data(raw_data):
    prep_data = raw_data['points'][0]['customdata']
    prep_data.pop('Unnamed: 0')
    prep_data.pop('y_est')
    prep_data['X'] = prep_data.pop('C1')
    prep_data['Y'] = prep_data.pop('C2')
    prep_data['Z'] = prep_data.pop('C3')
    prep_data['Fraud probability'] = prep_data.pop('y_probability')
    prep_data['Actually fraud'] = 'Yes' if prep_data.pop('y_true') == 1 else 'No'
    prep_data['Fraud amount'] = prep_data.pop('Fraud_amount')
    return prep_data

app.run_server(debug=True, port=8050)
