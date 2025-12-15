import numpy as np
from itertools import groupby

class HorizontalDensity():

    """Computes the horizontal densities of the chart by analyzing notes
    per second under weighted average mechanism. Also retrieves
    log(log(timestamp)) to address right skewed distribution of song lengths.

    `alpha` assigns more weight to larger note per seconds readings.
    `alpha = 0` is a vanilla average and `alpha > 0` is weighted using
    power sums. For best performance, use `alpha` between 0 and 3.
    """

    def __init__(self, alpha):
        self.alpha = alpha
        self.window_size = 1

    def compute(self, chart):
        horizontal_density = {}

        preprocessed = {k: sum(map(int, list(v)))
            for k, v in chart.items()}
        length = max(preprocessed.keys())
        generator = (zip(*g) for _, g in 
            groupby([(k // self.window_size * self.window_size, v)
            for k, v in preprocessed.items()]))
        notes_per_second = np.array(list({
            max(keys): sum(vals) for keys, vals in generator}.values()))

        horizontal_density['nps'] = np.sqrt(self._weighted_average(notes_per_second**2)) 
        horizontal_density['length'] = np.log(length)

        return horizontal_density

    def _weighted_average(self, values):
        weights = np.power(np.arange(len(values)), self.alpha)
        return np.dot(weights, np.sort(values))/np.sum(weights)
