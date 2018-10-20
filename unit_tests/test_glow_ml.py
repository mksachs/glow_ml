import os
import tempfile

import pytest
import pandas as pd

from glow_ml import glow_ml_app
import glow_ml.utilities as utils
import glow_ml.constants as const

test_data = [{
    'risk_score': 7.318181,
    'gender': 'M',
    'age': 30.0,
    'years_experience': 36.0,
    'role': 'Employee',
    'company': 89,
    'state': 2.0,
    'hours_worked_per_week': 36.914861
},
{
    'risk_score': 4.435424,
    'gender': 'M',
    'age': 24.0,
    'years_experience': 6.0,
    'role': 'Employee',
    'company': 41,
    'state': 4.0,
    'hours_worked_per_week': 57.901800
}]


@pytest.fixture
def client():
    # Prep a flask client to test against.
    glow_ml_app.config['TESTING'] = True
    client = glow_ml_app.test_client()

    yield client


@pytest.fixture
def model():
    # Load the active model directly and get predictions.
    pa_ml = utils.load_ml('{}/{}'.format(const.ML_ROOT, 'predict_accident.pkl'))
    yield pa_ml


def test_predict_accident(client, model):
    """Test the prediction API."""
    test_data_pd = pd.DataFrame.from_dict(test_data)
    pred = list(zip(model.predict(test_data_pd), model.predict_proba(test_data_pd)))

    # Get predictions from the API.
    pred_api_raw = client.post(
        '/api/v1.0/predict_accident',
        json=test_data,
        follow_redirects=True
    )
    pred_api = pred_api_raw.get_json()

    # These should both be the same.
    assert len(pred) == len(pred_api)

    for i in range(len(pred)):
        # Test the prediction
        assert pred[i][0] == pred_api[i][0]
        # Test the prediction probabilities.
        assert pred[i][1][0] == pred_api[i][1][0]
        assert pred[i][1][1] == pred_api[i][1][1]


def test_predict_accident_info(client, model):
    """Test the model info API."""
    ml_info = {
        'Best training score': model.best_score_,
        'Best training params': model.best_params_,
        'CV Results': model.cv_results_,
        'Params': model.get_params(deep=True)
    }

    # Get predictions from the API.
    ml_info_api_raw = client.get(
        '/api/v1.0/predict_accident/info',
        follow_redirects=True
    )
    ml_info_api = ml_info_api_raw.get_json()

    # These should both be the same.
    assert len(ml_info_api) == len(ml_info)

    for key in ml_info:
        assert key in ml_info_api

