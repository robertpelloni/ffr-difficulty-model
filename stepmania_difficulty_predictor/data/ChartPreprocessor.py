import numpy as np

class ChartPreprocessor():

    """Preprocesses FFR API response to a dictionary in following format:
    {
        'name': name of stepfile (str),
        'difficulty': manually assigned difficulty of stepfile (int),
        'chart': {
            timestamps (float): binary step encodings (str) 
        } 
    }
    """

    def __init__(self, decimals = 3):
        self.decimals = decimals
        self.mappings = {
            'L': 1000, 'D': 100,
            'U': 10, 'R': 1
        }

    def preprocess(self, chart):
        chart = np.roll(np.array(chart['chart']), 1, axis = 1)[:, :2]

        chart[:, 0] = chart[:, 0].astype(float)/1000.
        orientations, encodings = np.unique(chart[:, 1], return_inverse=True)
        chart[:, 1] = np.array([*map(self.mappings.get, orientations)])[encodings]
        chart = chart.astype(float)

        row_mask = np.append(np.diff(chart[:, 0], axis = 0) != 0, [True])
        cumsum_grps = chart.cumsum(0)[row_mask, 1:]
        sum_grps = np.diff(cumsum_grps, axis = 0)
        counts = np.concatenate((cumsum_grps[0, :][None], sum_grps), axis = 0)

        chart = np.concatenate((chart[row_mask, 0][:, None], counts), axis = 1)

        return {np.round(k - chart[:, 0].min(), self.decimals): str(int(v)).zfill(4) for k, v in chart}