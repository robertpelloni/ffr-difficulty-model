import numpy as np

class VerticalDensity():

    """Computes the vertical densities of the chart by analyzing timedeltas
    across multiple orientations under weighted harmonic mean mechanism.

    This implementation is mode-agnostic and will dynamically generate
    orientations based on the number of panels detected in the chart.

    `alpha` assigns more weight to smaller timedeltas. `alpha = 0` is
    a vanilla average and `alpha > 0` is weighted using power sums.
    For best performance, use `alpha` between 0 and 3.
    """

    def __init__(self, alpha):
        self.alpha = alpha

    def compute(self, chart):
        """
        Computes vertical density features for a given chart.
        """
        if not chart:
            return {}

        # Determine the number of panels from the first note's encoding
        num_panels = len(next(iter(chart.values()), []))
        if num_panels == 0:
            return {}

        # Define orientations dynamically using filter functions
        orientations = {}
        # Individual columns
        for i in range(num_panels):
            orientations[f'col_{i}'] = lambda v, i=i: v[i] == '1'

        # Halves (e.g., left vs right side of the pad)
        if num_panels > 1:
            left_half_cols = range(num_panels // 2)
            right_half_cols = range(num_panels // 2, num_panels)

            orientations['left'] = lambda v: any(v[i] == '1' for i in left_half_cols)
            orientations['right'] = lambda v: any(v[i] == '1' for i in right_half_cols)

        # Any note at all
        orientations['all'] = lambda v: '1' in v

        vertical_density = {}
        for orientation, filter_func in orientations.items():
            # Filter the chart to get timestamps for the current orientation
            filtered_keys = [k for k, v in chart.items() if filter_func(v)]

            if len(filtered_keys) < 2:
                vertical_density[orientation] = 0
                continue

            timedeltas = np.diff(np.array(sorted(filtered_keys)))
            density = self._weighted_harmonic_average(timedeltas)
            vertical_density[orientation] = density

        return vertical_density

    def _weighted_harmonic_average(self, values):
        """
        Calculates the weighted harmonic average of the given values.
        """
        # Filter out any zero or near-zero timedeltas to avoid division by zero
        values = values[values > 1e-6]
        if len(values) == 0:
            return 0

        weights = np.power(np.arange(len(values)), self.alpha)
        if np.sum(weights) > 0:
            # The harmonic mean gives more weight to smaller values
            return np.sum(weights) / np.dot(weights, np.reciprocal(np.sort(values)))
        else:
            return 0
