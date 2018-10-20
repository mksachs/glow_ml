from argparse import ArgumentParser
import os.path
import logging
from logging.handlers import RotatingFileHandler
import time
import warnings

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import sklearn.exceptions

import glow_ml_train.constants as const
from glow_ml_train.exceptions import AlgorithmNotImplemented
import glow_ml_train.utilities as utils
import glow_ml.constants as glow_ml_const

# Scikit-learn generates a ton of these warnings.
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=sklearn.exceptions.ConvergenceWarning)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = RotatingFileHandler('glow_ml_train.log', maxBytes=10000, backupCount=1)
handler.setLevel(logging.INFO)
logger.addHandler(handler)


def log_print(msg):
    """

    Args:
        msg:

    Returns:
        None:
    """
    print(msg)
    logger.info(msg)


def predict_accident(training_data, holdout=0, algorithm='lr'):
    """

    Args:
        training_data:
        holdout:
        algorithm:

    Returns:
        None:
    """
    if algorithm not in ('lr', 'gb', 'rf'):
        raise AlgorithmNotImplemented(f'The {algorithm} has not been implemented for "predict_accident".')

    log_print(f'Training "predict_accident" model using {training_data} with algorithm {algorithm} and a {holdout} '
              f'holdout set.'
              )

    raw_data = pd.read_csv(training_data)

    # Create the preprocessing pipelines for both numeric and categorical data.
    numeric_features = const.PREDICT_ACCIDENT_FEATURES['numeric']
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())])

    categorical_features = const.PREDICT_ACCIDENT_FEATURES['categorical']
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)])

    if algorithm == 'lr':
        clf = Pipeline(
            steps=[('preprocessor', preprocessor),
                   ('classifier', LogisticRegression(solver='saga'))
                   ])
        param_grid = {
            'classifier__C': [0.1, 1, 10],
            'classifier__max_iter': [10, 100, 500],
        }
    elif algorithm == 'gb':
        clf = Pipeline(
            steps=[('preprocessor', preprocessor),
                   ('classifier', GradientBoostingClassifier())
                   ])
        param_grid = {
            'classifier__learning_rate': [1.5, 1.7, 1.9],
            'classifier__n_estimators': [500, 750, 1000],
        }
    elif algorithm == 'rf':
        clf = Pipeline(
            steps=[('preprocessor', preprocessor),
                   ('classifier', RandomForestClassifier())
                   ])
        param_grid = {
            'classifier__n_estimators': [10, 50, 100],
        }

    X = raw_data.drop(const.PREDICT_ACCIDENT_TARGET, axis=1)
    y = raw_data[const.PREDICT_ACCIDENT_TARGET]

    if holdout != 0:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=holdout)
        test_data_dir = '/'.join(training_data.split('/')[0:-1])
        train_data_file_name = training_data.split('/')[-1].split('.')[0]
        test_data_path = f'{test_data_dir}/{train_data_file_name}_test.csv'
        log_print(f'Saving {holdout} testing data to {test_data_path}')
        test_data = pd.concat([X_test, y_test], axis=1)
        test_data.to_csv(test_data_path)
    else:
        X_train = X
        y_train = y

    log_print('Begin training.')
    start = time.time()

    grid = GridSearchCV(clf, param_grid, cv=5, iid=False, scoring='recall')
    grid.fit(X_train, y_train)

    end = (time.time() - start) / 60
    log_print(f'End training. Total time (minutes): {end}')

    log_print(f'Saving model to "{glow_ml_const.ML_ROOT}/predict_accident.pkl"')

    utils.save_ml(grid, f'{glow_ml_const.ML_ROOT}/predict_accident.pkl')


if __name__ == "__main__":

    parser = ArgumentParser(description='Train ML models to predict worker outcomes.')
    parser.add_argument(
        'model',
        choices=['predict_accident'],
        help='The model to train. Choices are "predict_accident".'
    )
    parser.add_argument(
        '-d', '--training_data',
        dest='training_data',
        help='The location of the training data to be used to train the model. '
             'If not specified this will default to "glow_ml_train/data/{model_type}.csv.'
    )
    parser.add_argument(
        '--algorithm',
        dest='algorithm',
        choices=['lr', 'rf', 'gb'],
        help='The type of algorithm to use to fit the data. The choices are: lr: Logistic Regression rf: Random Forest '
             'gb: Gradient Boosted Trees. A grid search over possible hyper-parameters of the algorithm will be '
             'performed. Default is "gb"',
        default='gb'
    )
    parser.add_argument(
        '--holdout',
        dest='holdout',
        help='A number between [0, 1) that specifies the percentage of the training data to be left out as a testing '
             'set. Default is 0 (ie. train on the entire set).',
        default=0,
        type=float
    )

    args = parser.parse_args()

    training_data_path = f'glow_ml_train/data/{args.model}.csv'
    if args.training_data:
        training_data_path = args.training_data

    if not os.path.isfile(training_data_path) :
        raise FileNotFoundError(f'{os.getcwd()}{training_data_path} does not exist.')

    result = globals()[args.model](training_data_path, holdout=args.holdout, algorithm=args.algorithm)