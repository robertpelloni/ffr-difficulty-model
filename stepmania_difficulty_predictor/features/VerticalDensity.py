import numpy as np
import re

class VerticalDensity():

    """Computes the vertical densities of the chart by analyzing timedeltas
    across multiple orientations under weighted harmonic mean mechanism

    `alpha` assigns more weight to smaller timedeltas. `alpha = 0` is
    a vanilla average and `alpha > 0` is weighted using power sums.
    For best performance, use `alpha` between 0 and 3.
    """

    def __init__(self, alpha):
        self.alpha = alpha
        self.regex = {
            'L': r'1...', 'D': r'.1..',
            'U': r'..1.', 'R': r'...1',
            'left': r'1...|.1..|11..',
            'right': r'...1|..1.|..11',
            'all': r'....'
        }

    def compute(self, chart):
        vertical_density = {}
        for orientation in self.regex.keys():
            filtered = {
                k : v for k, v in chart.items() 
                if re.match(self.regex[orientation], v)
            }
            timedeltas = np.diff(np.array(list(filtered.keys())))
            density = self._weighted_harmonic_average(timedeltas)
            vertical_density[orientation] = density
        return vertical_density

    # def _weighted_average(self, values):
    #     weights = np.power(np.arange(len(values)), self.alpha)
    #     return np.dot(weights, np.sort(values)[::-1])/np.sum(weights)

    def _weighted_harmonic_average(self, values):
        weights = np.power(np.arange(len(values)), self.alpha)
        if np.sum(weights) != 0:
            return np.dot(weights, np.reciprocal(np.sort(values)[::-1]))/np.sum(weights)
        else:
            return 0