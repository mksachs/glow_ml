import logging
from logging.handlers import RotatingFileHandler
import warnings

warnings.filterwarnings('ignore', category=DeprecationWarning)

from flask import Flask

from glow_ml.utilities import GlowMLJSONEncoder

glow_ml_app = Flask(__name__)
glow_ml_app.json_encoder = GlowMLJSONEncoder
handler = RotatingFileHandler('glow_ml.log', maxBytes=10000, backupCount=1)
handler.setLevel(logging.INFO)
glow_ml_app.logger.addHandler(handler)

import glow_ml.views.api
