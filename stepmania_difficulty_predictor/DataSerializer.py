import os
import pickle

class DataSerializer():

    """Converts generated data structure into serialized .chart file
    under naming convention {id}.chart

    Saves converted results in parameter `folder`
    """

    def __init__(self, folder):
        self.folder = folder

    def download(self, info, id):
        filename = f"{self.folder}/{str(id).zfill(4)}.chart"
        if os.path.exists(filename):
            old_info = pickle.load( open( filename, "rb" ) )
            if info == old_info:
                return
        pickle.dump(info, open( filename, "wb" ) )




