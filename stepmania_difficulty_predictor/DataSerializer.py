import os
import json
import numpy as np

class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

class DataSerializer():

    """Converts generated data structure into a JSON-serialized .chart file
    under the naming convention {id}.chart.

    Saves converted results in the parameter `folder`.
    """

    def __init__(self, folder):
        self.folder = folder

    def download(self, info, id):
        """
        Serializes the chart information to a JSON file.
        """
        filename = f"{self.folder}/{str(id).zfill(4)}.chart"

        # We always overwrite now since JSON is deterministic and fast
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(info, f, cls=NumpyEncoder, ensure_ascii=False, indent=4)
