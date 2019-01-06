# -*- coding: utf-8 -*-

import pandas as pd
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go

from dataset_utils import *

# colorscale for probability colorbas
colorscale = [[0.0, 'rgb(0,51,255)'],
              [0.1, 'rgb(71,108,255)'],
              [0.2, 'rgb(104,134,255)'],
              [0.3, 'rgb(152,173,255)'],
              [0.4, 'rgb(255,142,142)'],
              [1.0, 'rgb(255,0,0)'],
              ]

# Initial dataset
DATA_DIR_PATH = os.path.join(os.path.curdir, 'data/')
available_datasets = find_csv_filenames(DATA_DIR_PATH)

# Initial graph params
graph_layout = dict(
    # autosize=True,
    legend=dict(x=-0.1, y=1.2),
    margin=dict(
        l=0,
        r=0,
        b=50,
        t=0
    ),
    uirevison=True
)

# Instantiate server
app = dash.Dash("Fraud Data Explorer")

# Main layout
app.layout = html.Div([
    # Hidden in-browser variables
    html.Div(hidden=True, id='available-datasets-names'),
    html.Div(hidden=True, id='current-dataset-name'),

    html.H2('Fraud Data Explorer'),
    html.Div(className='row', children=[
        # Graph
        html.Div([
            html.Label("Visual representation in 3D Space"),
            html.Div(className='nine columns', children=[
                dcc.Graph(
                    id='graph',
                    figure=dict(
                        layout=graph_layout,
                    ),
                ),
            ], id="graph-container")]),

        # Marker details
        html.Div(className='two columns', children=[
            html.Label("Marker details"),
            html.Div(id='marker-info', children=[
            ])
        ])

    ], style={'border-style': 'groove'}),

    html.Div(className='row', children=[

        html.Div(className='five columns', children=[
            # With or without nonfrauds
            dcc.Checklist(
                id='nonfrauds',
                options=[
                    {'label': 'With Nonfrauds', 'value': 'Yes'},
                ],
                values=['Yes'],

            ),

            dcc.RadioItems(
                options=[
                    {'label': 'Threshold', 'value': 'threshold'},
                    {'label': 'Actual results', 'value': 'actual'},
                    {'label': 'Default estimate', 'value': 'default'}],
                id='mode-selector',
                value='default',
                labelStyle={'display': 'inline-block'}
            ),

            html.Div([
                html.Div(id="threshold-selected"),

                html.Div([
                    dcc.Slider(
                        id='threshold',
                        min=0.01,
                        max=1.0,
                        step=0.01,
                        value=0.5,
                    ),
                ]),

                dcc.Checklist(
                    id='mistakes',
                    options=[
                        {'label': 'Highlight mistakes', 'value': 'Yes'},
                    ],
                    values=[]
                )],
                id='threshold-selector', hidden=True),

            # Choosing dataset
            html.Label('Dataset'),
            dcc.Dropdown(
                id='dataset',
                options=[{'label': i, 'value': i} for i in available_datasets],
                value=available_datasets[0],
            ),
        ]),

        # Confusion matrix overview
        html.Div(id='class-info', className='three columns', children=[]),

    ])
])

"""
Dataset choose callbacks
"""


# Update internal name of the chosen dataset
@app.callback(Output('current-dataset-name', 'children'),
              [Input('dataset', 'value')])
def update_current_dataset_name(name):
    print(name)
    return name


# When train set is chosen, mode selector must be set to actual and disabled
@app.callback(Output('mode-selector', 'options'),
              [Input('current-dataset-name', 'children')])
def disable_mode_selector_when_trainset(name):
    opts = [
        {'label': 'Threshold', 'value': 'threshold'},
        {'label': 'Actual results', 'value': 'actual'},
        {'label': 'Default estimate', 'value': 'default'}
    ]
    for opt in opts: opt.update({'disabled': 'true'}) if 'train' in name else None
    return opts


@app.callback(Output('mode-selector', 'value'),
              [Input('current-dataset-name', 'children')])
def set_actual_mode_when_trainset(name):
    if 'train' in name:
        return 'actual'
    else:
        return 'default'


""" 
Threshold choose callbacks
"""


@app.callback(Output('threshold-selector', 'hidden'),
              [Input('mode-selector', 'value')],
              [State('current-dataset-name', 'children')])
def show_hide_threshold_when_mode_chosen(selected_mode, dataset_name):
    if selected_mode == 'threshold' and 'train' not in dataset_name:
        return False
    else:
        return True


@app.callback(Output('threshold-selected', 'children'),
              [Input('threshold', 'value')])
def update_slider_info(value):
    return "Selected threshold: {}".format(value)


"""
Graph update callbacks
"""


@app.callback(Output('graph-container', 'children'),
              [Input('dataset', 'value'),
               Input('mode-selector', 'value'),
               Input('threshold', 'value'),
               Input('nonfrauds', 'values'),
               Input('mistakes', 'values')],
              [State('graph', 'relayoutData')]
              )
def update_graph(dataset, mode, threshold, with_non_frauds, mistakes, relayout_data):
    # Keeping the zoom locked

    # Read the data
    new_data = pd.read_csv(DATA_DIR_PATH + dataset)

    if mode == 'default':
        data = get_simple_markers(new_data,
                                  with_non_frauds=len(with_non_frauds) != 0,
                                  default=True)
    elif mode == 'actual':
        data = get_simple_markers(new_data,
                                  with_non_frauds=len(with_non_frauds) != 0,
                                  default=False)
    else:
        data = get_markers(new_data,
                           threshold=threshold,
                           with_non_frauds=len(with_non_frauds) != 0,
                           mistakes=len(mistakes) != 0)
    # Updating the figure
    result = dcc.Graph(
        id='graph',
        figure=dict(
            data=data,
            layout=graph_layout,
        ),
        relayoutData=relayout_data
    ),

    return result


"""
Classifier info update callbacks
"""


@app.callback(Output('class-info', 'children'),
              [Input('dataset', 'value'),
               Input('mode-selector', 'value'),
               Input('threshold', 'value')])
def update_classifier_info(dataset, mode, threshold):
    data = pd.read_csv(os.path.join(DATA_DIR_PATH, dataset))
    if mode == 'threshold':
        matrix = get_shapes_of_conf_mat(get_confusion_matrix(data, threshold))
    elif mode == 'default':
        matrix = get_shapes_of_conf_mat(get_confusion_matrix(data, threshold=0.01, default_estimation=True))
    else:
        matrix = {}

    return html.Table(
        [html.Tr([html.Th(column) for column in matrix.keys()])] +
        [html.Tr([html.Td(value) for value in matrix.values()])]
    )


"""
Marker info update callbacks
"""


@app.callback(Output('marker-info', 'children'),
              [Input('graph', 'clickData')],
              [State('current-dataset-name', 'children')])
def update_marker_info(selected_data, current_dataset):
    if selected_data is None:
        return ''
    # Get marker's custom data and clean it up
    marker_info = cleanup_marker_data(selected_data, 'train' in current_dataset)
    return html.Table(
        [html.Tr([html.Th(key), html.Td(value)]) for key, value in zip(marker_info.keys(), marker_info.values())])


def cleanup_marker_data(raw_data, train=False):
    """
    Cleans up raw custom data of the marker

    :rtype: dict
    :param raw_data:
    :param train: if True, info for train set will be returned
    :return: cleaned-up dictionary with parameters of the transaction
    """

    prep_data = raw_data['points'][0]['customdata']
    prep_data.pop('Unnamed: 0')

    prep_data['X'] = prep_data.pop('C1')
    prep_data['Y'] = prep_data.pop('C2')
    prep_data['Z'] = prep_data.pop('C3')

    if train:
        prep_data['Is fraud'] = 'Yes' if prep_data.pop('y') == 1.0 else 'No'
    else:
        prep_data.pop('y_est')
        prep_data['Fraud probability'] = '{:.2f}'.format(prep_data.pop('y_probability'))
        prep_data['Is fraud'] = 'Yes' if prep_data.pop('y_true') == 1 else 'No'
        prep_data['Mistake'] = prep_data.pop('Mistake') if 'Mistake' in prep_data else 'No'

    prep_data['Fraud amount'] = prep_data.pop('Fraud_amount')
    return prep_data


def get_scatter_3d(df, name, is_frauds, mistakes=False, false_negative=False):
    """
    Returns a scatter based on parameters

    :param df: DataFrame
    :param name: name of the trace
    :param is_frauds: dataframe consists frauds or not
    :param mistakes: is dataframe a mistake set
    :param false_negative: are mistakes false_negatives (to define color)
    :return: go.Scatter3d
    """

    # Color mistakes: FN - black, FP - yellow and adds additional field to custom data
    if mistakes:
        if false_negative:
            color_info = {'color': 'black'}
            mistake_info = 'False Negative'
        else:
            color_info = {'color': 'gold'}
            mistake_info = 'False Positive'
    else:
        # Implicit checking, if the dataset is train or test
        color_info = dict(color='red') if 'y_probability' not in df.columns else dict(cmin=0.0, cmax=1.0,
                                                                                      color=df['y_probability'],
                                                                                      colorscale=colorscale,
                                                                                      colorbar=dict(
                                                                                          x=-0.1, y=0.45,
                                                                                          title='Probability'))
        mistake_info = ''

    # Style the traces, basing on frauds/non-frauds
    if is_frauds:
        marker_style = dict(
            size=abs(3 + df["Fraud_amount"] / 6000),
            line=dict(
                color='white',
                width=0.5
            ),
            opacity=0.8
            , **color_info)
    else:
        marker_style = dict(
            size=2,
            color='grey',
            line=dict(
                color='white',
                width=0.5
            ),
            opacity=0.8
        )
    # Add custom data for later showing
    custom_data = df.to_dict('records')
    if mistake_info != '':
        for row in custom_data:
            row['Mistake'] = mistake_info
    return go.Scatter3d(
        x=df['C1'],
        y=df['C2'],
        z=df['C3'],
        name=name,
        customdata=custom_data,
        mode='markers',
        marker=marker_style
    )


def get_simple_markers(data, with_non_frauds=True, default=True):
    """

    Return markers for the graph from default estimation or actual results

    :param data: DataFrame dataset
    :param with_non_frauds: include non-frauds
    :param default: use default estimation
    :return: markers traces
    """
    info = get_classificator_info(data)

    frauds = info['default_frauds'] if default else info['act_frauds']
    trace_frauds = get_scatter_3d(frauds,
                                  'Default estimate frauds' if default else 'Actual frauds',
                                  is_frauds=True)

    nonfrauds = info['default_non_frauds'] if default else info['act_non_frauds']
    trace_nonfrauds = get_scatter_3d(nonfrauds,
                                     'Default estimate non-frauds' if default else 'Actual non-frauds',
                                     is_frauds=False)

    return [trace_frauds] if not with_non_frauds else [trace_frauds, trace_nonfrauds]


def get_markers(data, threshold, with_non_frauds=True, mistakes=False):
    """
    Return scatters based on parameters

    :param data: DataFrame dataset
    :param threshold: threshold [0.01 <= x <= 1.0]
    :param with_non_frauds: include non-frauds
    :param mistakes: adds mistakes (FN, FP) traces
    :rtype: list of go.Scatter3d
    :return: 3d scatter plots
    """
    info = get_classificator_info(data, threshold)
    if mistakes:
        matrix = get_confusion_matrix(data, threshold)
        trace_fn = get_scatter_3d(matrix['fn'], 'False negative', True, mistakes=True, false_negative=True)
        trace_fp = get_scatter_3d(matrix['fp'], 'False positive', True, mistakes=True, false_negative=False)
        trace_tp = get_scatter_3d(matrix['tp'], 'Predicted Frauds', True)
        trace_tn = get_scatter_3d(matrix['tn'], 'Predicted Non-frauds', False)
        return [trace_fn, trace_fp, trace_tn, trace_tp] if with_non_frauds else [trace_fn, trace_fp, trace_tp]
    else:
        frauds = info['pred_frauds']
        nonfrauds = info['pred_non_frauds']
        trace_frauds = get_scatter_3d(frauds, 'Predicted Frauds', is_frauds=True)
        trace_nonfrauds = get_scatter_3d(nonfrauds, 'Predicted Non-frauds', is_frauds=False)
        return [trace_frauds] if not with_non_frauds else [trace_frauds, trace_nonfrauds]


app.run_server(debug=True, port=8050)
