import numpy as np

class PatternDetector:
    """
    Detects and quantifies various patterns in a chart, such as jacks and crossovers.
    """
    def __init__(self, jack_threshold=0.1):
        """
        Initializes the PatternDetector.

        Args:
            jack_threshold: The maximum time between notes to be considered a jack.
        """
        self.jack_threshold = jack_threshold

    def compute(self, chart: dict) -> dict:
        """
        Computes the pattern features for a given chart.

        Args:
            chart: A dictionary representing the chart, with timestamps as keys
                   and binary step encodings as values.

        Returns:
            A dictionary containing the pattern features.
        """
        timestamps = sorted(chart.keys())
        if len(timestamps) < 2:
            return {'jack_percentage': 0, 'crossover_percentage': 0}

        jacks = 0
        crossovers = 0

        # Track the last note for jack detection
        last_notes = [-1, -1, -1, -1]

        for i in range(1, len(timestamps)):
            time_diff = timestamps[i] - timestamps[i-1]
            note = chart[timestamps[i]]
            prev_note = chart[timestamps[i-1]]

            # Jack detection
            if time_diff <= self.jack_threshold:
                for j in range(4):
                    if note[j] == '1' and prev_note[j] == '1':
                        jacks += 1

            # Crossover detection (simple version)
            # Left foot (Down/Left) and Right foot (Up/Right)
            left_foot_on_right = (note[3] == '1' or note[2] == '1') and \
                                 (prev_note[0] == '1' or prev_note[1] == '1')
            right_foot_on_left = (note[0] == '1' or note[1] == '1') and \
                                 (prev_note[3] == '1' or prev_note[2] == '1')

            if left_foot_on_right or right_foot_on_left:
                crossovers += 1

        total_notes = len(timestamps)
        jack_percentage = (jacks / total_notes) * 100 if total_notes > 0 else 0
        crossover_percentage = (crossovers / total_notes) * 100 if total_notes > 0 else 0

        return {
            'jack_percentage': jack_percentage,
            'crossover_percentage': crossover_percentage
        }
