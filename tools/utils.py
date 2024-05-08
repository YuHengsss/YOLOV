import pickle
import json
import os

def save_pkl(filename, data):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)


def load_pkl(name):
    return pickle.load(open(name, 'rb'))

def load_json(name):
    return json.load(open(name, 'r'))


def save_json(filename, data):
    with open(filename, 'w') as f:
        json.dump(data, f)


def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)  # mp



