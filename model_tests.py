from argparse import ArgumentParser
import os.path
import warnings
import requests

import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, roc_auc_score,
    precision_recall_curve, roc_curve
)
import matplotlib; matplotlib.use('AGG')
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

import glow_ml_train.constants as const
import glow_ml.utilities as glow_ml_utils
import glow_ml.constants as glow_ml_const

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)


def predict(model, data):
    clf = glow_ml_utils.load_ml(f'{glow_ml_const.ML_ROOT}/{model}.pkl')
    return clf.predict(data), clf.predict_proba(data)


def predict_remote(model, data, url):
    data_list = data.to_dict('records')
    results = requests.post(url, json=data_list)
    pred = []
    proba = []
    for item in results.json():
        pred.append(item[0])
        proba.append(item[1])
    return np.array(pred), np.array(proba)


def output_metrics(predictions, probabilities, truth):
    accuracy = accuracy_score(truth, predictions)
    precision = precision_score(truth, predictions)
    recall = recall_score(truth, predictions)
    f1 = f1_score(truth, predictions)

    cm = confusion_matrix(truth, predictions)

    tn, fp, fn, tp = cm.ravel()

    true_negative_rate = tn / (tn + fp)
    negative_predictive_value = tn / (fn + tn)
    false_negative_rate = 1 - recall
    false_positive_rate = 1 - true_negative_rate
    false_discovery_rate = 1 - precision
    false_ommision_rate = 1 - negative_predictive_value

    print('Confusion Matrix')
    print(f'|{tp:^10}|{fp:^10}|')
    print(f'|{fn:^10}|{tn:^10}|')

    print(f'Accuracy: {accuracy}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1: {f1}')
    print(f'True Negative Rate: {true_negative_rate}')
    print(f'Negative Predictive Value: {negative_predictive_value}')
    print(f'False Positive Rate: {false_positive_rate}')
    print(f'False Negative Rate: {false_negative_rate}')
    print(f'False Discovery Rate: {false_discovery_rate}')
    print(f'False Omission Rate: {false_ommision_rate}')


def plot_pr_curve(truth, proba, output, size, placement, model_name):
    fig = plt.figure(figsize=size)
    ax = fig.add_axes(placement)

    precision, recall, _ = precision_recall_curve(truth, proba)

    ax.step(recall, precision, color='b', alpha=0.2, where='post')
    ax.fill_between(recall, precision, alpha=0.2, color='b', step='post')

    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_ylim([0.0, 1.05])
    ax.set_xlim([0.0, 1.0])
    ax.set_title(f'{model_name} Precision-Recall curve')

    print(f'Precision-Recall curve saved to {output}')

    fig.savefig(output)


def plot_roc_curve(truth, proba, output, size, placement, model_name):
    fig = plt.figure(figsize=size)
    ax = fig.add_axes(placement)

    fpr, tpr, _ = roc_curve(truth, proba)
    auc = roc_auc_score(truth, proba)

    ax.plot(fpr, tpr, color='darkorange', label=f'ROC curve (area = {auc})')
    ax.plot([0, 1], [0, 1], color='navy', linestyle='--')

    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_ylim([0.0, 1.05])
    ax.set_xlim([0.0, 1.0])
    ax.legend(loc='lower right')
    ax.set_title(f'{model_name} Receiver Operating Characteristic')

    print(f'Receiver Operating Characteristic saved to {output}')

    fig.savefig(output)


def plot_class_curve(truth, pred, proba, output, size, placement, model_name):
    fig = plt.figure(figsize=size)
    ax = fig.add_axes(placement)

    data = pd.DataFrame(
        {'truth': truth, 'pred': pred, 'proba_true': proba[:, 1], 'proba_false': proba[:, 0]},
        columns=['truth', 'pred', 'proba_true', 'proba_false']
    )

    ax = sns.distplot(
        data.query('truth == 0')['proba_true'],
        ax=ax,
        hist=False,
        label='False'
    )
    ax = sns.distplot(
        data.query('pred == 0')['proba_true'],
        ax=ax,
        hist=False,
        label='Predicted False'
    )

    ax = sns.distplot(
        data.query('truth == 1')['proba_true'],
        ax=ax,
        hist=False,
        label='True',
        kde_kws={'ls': '--'}
    )
    ax = sns.distplot(
        data.query('pred == 1')['proba_true'],
        ax=ax,
        hist=False,
        label='Predicted True',
        kde_kws={'ls': '--'}
    )

    ax.set_xlabel('Probability True')
    ax.set_ylabel('Density')

    ax.legend()
    ax.set_title(f'{model_name} Class Prediction Distribution')

    print(f'Class Prediction Distribution saved to {output}')

    fig.savefig(output)


def output_curves(predictions, probabilities, truth, output_path, model_name):
    sns.set()

    figsize = (800 / 72, 600 / 72)
    figplacement = [.1, .1, .8, .8]

    model_file_name = '_'.join(model_name.lower().split(' '))

    plot_pr_curve(
        truth, probabilities[:,0],
        f'{output_path}/{model_file_name}_pr_curve.png',
        figsize, figplacement, model_name
    )

    plot_roc_curve(
        truth, probabilities[:, 0],
        f'{output_path}/{model_file_name}_roc_curve.png',
        figsize, figplacement, model_name
    )

    plot_class_curve(
        truth, predictions, probabilities,
        f'{output_path}/{model_file_name}_class_curve.png',
        figsize, figplacement, model_name
    )


def predict_accident(test_data, test_url=None):
    raw_data = pd.read_csv(test_data)

    X = raw_data.drop(const.PREDICT_ACCIDENT_TARGET, axis=1)
    y = raw_data[const.PREDICT_ACCIDENT_TARGET]

    if test_url is None:
        predictions, probabilities = predict('predict_accident', X)
    else:
        predictions, probabilities = predict_remote('predict_accident', X, test_url)

    print('### Stats for the "Predict Accident" model ###')

    output_metrics(predictions, probabilities,  y)

    test_data_dir = '/'.join(test_data.split('/')[0:-1])
    output_curves(predictions, probabilities,  y, test_data_dir, 'Predict Accident')


if __name__ == "__main__":

    parser = ArgumentParser(description='Evaluate model performance using holdout test data sets.')
    parser.add_argument(
        'model',
        choices=['predict_accident'],
        help='The model to test. Choices are "predict_accident".'
    )
    parser.add_argument(
        '-d', '--test_data',
        dest='test_data',
        help='The location of the test data to be used to test the model. '
             'If not specified this will default to "glow_ml_train/data/{model_type}_test.csv.'
    )
    parser.add_argument(
        '--test_url',
        dest='test_url',
        help='The url of the model api to test. If not specified the model will be deserialized in the testing module '
             'and tested there.'
    )

    args = parser.parse_args()

    test_data_path = f'glow_ml_train/data/{args.model}_test.csv'
    if args.test_data:
        test_data_path = args.test_data

    if not os.path.isfile(test_data_path) :
        raise FileNotFoundError(f'{os.getcwd()}{test_data_path} does not exist.')

    result = globals()[args.model](test_data_path, test_url=args.test_url)