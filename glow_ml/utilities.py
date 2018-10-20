import pickle

from flask.json import JSONEncoder
import numpy as np


def load_ml(path):
    return pickle.load(open(path, 'rb'))


class GlowMLJSONEncoder(JSONEncoder):
    def default(self, obj):
        try:
            if (
                    isinstance(obj, np.int_)
                    or isinstance(obj, np.int32)
                    or isinstance(obj, np.int64)
            ):
                if np.isnan(obj):
                    return None
                return int(obj)
            elif (
                    isinstance(obj, np.float_)
                    or isinstance(obj, np.float32)
                    or isinstance(obj, np.float64)
            ):
                if np.isnan(obj):
                    return None
                return float(obj)
            elif isinstance(obj, np.bool_):
                if np.isnan(obj):
                    return None
                return bool(obj)

            iterable = iter(obj)
        except (TypeError, AttributeError):
            return str(obj)
        else:
            return list(iterable)