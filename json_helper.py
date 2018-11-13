import _pickle as cpickle
import pickle
import json


def deserialize_from_file(filename="data.json"):
    with open(filename, "rb") as read_file:
        data = json.load(read_file)
        return data


def serialize_to_file(filename="data.json", data=None):
    with open(filename, "w") as write_file:
        json.dump(data, write_file)


