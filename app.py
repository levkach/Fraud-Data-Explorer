# -*- coding: utf-8 -*-

import pandas as pd
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import os
import colorlover as cl


# Finds csv files
def find_csv_filenames(path_to_dir, suffix=".csv"):
    filenames = os.listdir(path_to_dir)
    return [filename for filename in filenames if filename.endswith(suffix)]


# Initial dataset
DATA_DIR_PATH = os.path.join(os.path.curdir, 'data/')
avaliable_data_sets = find_csv_filenames(DATA_DIR_PATH)
raw_data = pd.read_csv("data/PCA_test_set.csv")

# Instantiate server
app = dash.Dash("Fraud Data Explorer")


# Takes raw data, produces frauds, nonfrauds based on given threshold or actually from the data.
def get_classificator_info(data, threshold=0.5):
    """
    :param data:  dataset
    :param threshold:  threshold for decision making
    :return: dictionary of actual and predicted frauds of the dataset
    :rtype: dict
    """

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

    # Default results
    default_frauds = data[data.y_est == 1.0]
    default_non_frauds = data[data.y_est == 0.0]

    ret = {'act_frauds': actual_frauds, 'act_non_frauds': actual_non_frauds,
           'pred_frauds': predicted_frauds, 'pred_non_frauds': predicted_non_frauds,
           'default_frauds': default_frauds, 'default_non_frauds': default_non_frauds}
    return ret


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
        color_info = dict(cmin=-2.0, cmax=1.0,
                          color=df['y_probability'],
                          colorscale=[[x / 10, y] for x in range(0, 10, 2) for y in
                                      cl.to_rgb(cl.scales['5']['seq']['Reds'])])
        mistake_info = ''

    # Style the traces, basing on frads/non-frauds
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
    # Add custom data for later show
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


def get_confusion_matrix(data, threshold, default_estimation=False):
    """

    :param data: DataFrame dataset
    :param threshold: [0.01 <= x <= 1.0]
    :param default_estimation: use default dataset's estimation instead of threshold
    :return: dict of confusion matrix
    """
    # Calculate margins
    fn = data[
        (data.y_true == 1.0) & ((data.y_probability < threshold) if not default_estimation else (data.y_est == 0.0))]
    fp = data[
        (data.y_true == 0.0) & ((data.y_probability >= threshold) if not default_estimation else (data.y_est == 1.0))]
    tn = data[
        (data.y_true == 0.0) & ((data.y_probability < threshold) if not default_estimation else (data.y_est == 0.0))]
    tp = data[
        (data.y_true == 1.0) & ((data.y_probability >= threshold) if not default_estimation else (data.y_est == 0.0))]

    precision = 1.0 * tp.shape[0] / (tp.shape[0] + fp.shape[0])
    recall = 1.0 * tp.shape[0] / (tp.shape[0] + fn.shape[0])

    return {'fn': fn, 'fp': fp, 'tn': tn, 'tp': tp, 'precision': precision, 'recall': recall}


# Initial graph params
graph_layout = dict(
    autosize=True,
    legend=dict(x=-0.1, y=1.2),
    margin=dict(
        l=0,
        r=0,
        b=50,
        t=0
    )
)

# Main layout
app.layout = html.Div([
    # Hidden in-browser variables
    html.Div(hidden=True, id='available-datasets-names'),
    html.Div(hidden=True, id='current-dataset-name'),
    # dcc.Store(storage_type="local", id="current-dataset"),

    html.H2('Fraud Data Explorer'),
    html.Div(className='row', children=[
        # Graph
        html.Div([
            html.Label("Visual representation in 3D Space"),
            html.Div(className='nine columns', children=[
                dcc.Graph(
                    id='graph',
                    figure=dict(
                        data=get_simple_markers(raw_data),
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
            html.Label('Non-frauds'),
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
                )], id='threshold-selector', hidden=True
            ),

            # Choosing dataset
            html.Label('Dataset'),
            dcc.Dropdown(
                id='dataset',
                options=[{'label': i, 'value': i} for i in avaliable_data_sets],
                value=avaliable_data_sets[0],
            ),
        ]),

        # Confusion matrix overview
        html.Div(id='class-info', className='three columns', children=[]),

    ])
])


# # Dataset choose callbacks
@app.callback(Output('current-dataset-name', 'children'),
              [Input('dataset', 'value')])
def update_current_dataset_name(name):
    return name


# @app.callback(Output('current-dataset', 'data'),
#               [Input('current-dataset-name', 'children')])
# def update_current_dataset_value(new_name):
#     return pd.read_csv(DATA_DIR_PATH + "/" + new_name).to_dict('records')


# Threshold choose callbacks
@app.callback(Output('threshold-selector', 'hidden'),
              [Input('mode-selector', 'value')])
def update_mode_selector(selected_mode):
    if selected_mode == 'threshold':
        return False
    else:
        return True


@app.callback(Output('threshold-selected', 'children'),
              [Input('threshold', 'value')])
def update_slider(value):
    return "Selected threshold: {}".format(value)


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
    print(relayout_data)

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


@app.callback(Output('class-info', 'children'),
              [Input('dataset', 'value'),
               Input('mode-selector', 'value'),
               Input('threshold', 'value')])
def update_classificator_info(dataset, mode, threshold):
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


def get_shapes_of_conf_mat(matrix):
    result = {
        'True Positive': matrix['tp'].shape[0],
        'True Negative': matrix['tn'].shape[0],
        'False Positive': matrix['fp'].shape[0],
        'False Negative': matrix['fn'].shape[0],
        'precision': '{:.3f}'.format(matrix['precision']),
        'recall': '{:.3f}'.format(matrix['recall'])
    }
    return result


@app.callback(Output('marker-info', 'children'),
              [Input('graph', 'clickData')])
def update_marker_info(selected_data):
    if selected_data is None:
        return ''
    # Get marker's custom data and clean it up
    marker_info = cleanup_marker_data(selected_data)
    return html.Table(
        [html.Tr([html.Th(key), html.Td(value)]) for key, value in zip(marker_info.keys(), marker_info.values())])


def cleanup_marker_data(raw_data):
    """
    Cleans up raw data from the marker

    :rtype: dict
    :param raw_data:
    :return: cleaned-up dictionary with parameters of the case
    """
    prep_data = raw_data['points'][0]['customdata']
    prep_data.pop('Unnamed: 0')
    prep_data.pop('y_est')
    prep_data['X'] = prep_data.pop('C1')
    prep_data['Y'] = prep_data.pop('C2')
    prep_data['Z'] = prep_data.pop('C3')
    prep_data['Fraud probability'] = '{:.2f}'.format(prep_data.pop('y_probability'))
    prep_data['Is fraud'] = 'Yes' if prep_data.pop('y_true') == 1 else 'No'
    prep_data['Fraud amount'] = prep_data.pop('Fraud_amount')
    prep_data['Mistake'] = prep_data.pop('Mistake') if 'Mistake' in prep_data else 'No'
    return prep_data


app.run_server(debug=True, port=8050)
