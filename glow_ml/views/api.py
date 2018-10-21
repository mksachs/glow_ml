import flask
import pandas as pd

from glow_ml import glow_ml_app
import glow_ml.utilities as utils
import glow_ml.constants as const

pa_ml = utils.load_ml('{}/{}'.format(const.ML_ROOT, 'predict_accident.pkl'))


@glow_ml_app.route('/api/v1.0/predict_accident', methods=['POST'])
def predict_accident():

    data = pd.DataFrame.from_dict(flask.request.get_json())
    predictions = pa_ml.predict(data)
    probabilities = pa_ml.predict_proba(data)

    predictions_probabilities = list(zip(predictions, probabilities))

    # Logging example. This should probably be emitted to something like InfluxDB.
    glow_ml_app.logger.info(f'predictions: {predictions_probabilities}')

    return flask.jsonify(predictions_probabilities)


@glow_ml_app.route('/api/v1.0/predict_accident/info', methods=['GET'])
def predict_accident_info():

    ml_info = {
        'Best training score': pa_ml.best_score_,
        'Best training params': pa_ml.best_params_,
        'CV Results': pa_ml.cv_results_,
        'Params': pa_ml.get_params(deep=True)
    }

    return flask.jsonify(ml_info)
