"""

dataset_utils.py contains functions required for retrieving information about given dataset

"""


def get_classificator_info(data, threshold=0.5):
    """

    Takes raw data, produces frauds, nonfrauds based on given threshold or actually from the data.

    :param data:  dataset
    :param threshold:  threshold for decision making
    :return: dictionary of actual and predicted frauds of the dataset
    :rtype: dict
    """

    # Train set.
    is_train_set = len(data.columns) == 6
    print(len(data.columns))
    # Actual results
    actual_frauds = data[data.y == 1.0 if is_train_set else data.y_true == 1.0]
    actual_non_frauds = data[data.y == 0 if is_train_set else data.y_true == 0.0]

    ret = {'act_frauds': actual_frauds, 'act_non_frauds': actual_non_frauds}

    if not is_train_set:
        # Predicted results
        predicted_frauds = data[data.y_probability >= threshold]
        predicted_non_frauds = data[data.y_probability < threshold]

        # Default results
        default_frauds = data[data.y_est == 1.0]
        default_non_frauds = data[data.y_est == 0.0]
        ret.update({'pred_frauds': predicted_frauds, 'pred_non_frauds': predicted_non_frauds,
                    'default_frauds': default_frauds, 'default_non_frauds': default_non_frauds})

    return ret


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
