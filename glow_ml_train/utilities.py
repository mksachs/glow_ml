import pickle


def save_ml(obj, path):
    return pickle.dump(obj, open(path, 'wb'))