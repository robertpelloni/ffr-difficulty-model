import numpy as np

class PatternDetector:
    """
    Detects and quantifies various patterns in a chart, such as jacks and crossovers.
    This implementation is mode-agnostic and will adapt to the number of panels
    detected in the chart.
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

        # Determine the number of panels from the first note's encoding
        num_panels = len(next(iter(chart.values()), []))
        if num_panels == 0:
            return {'jack_percentage': 0, 'crossover_percentage': 0}

        jacks = 0
        crossovers = 0

        last_note_com = 0 # Center of mass for the previous note

        for i in range(1, len(timestamps)):
            time_diff = timestamps[i] - timestamps[i-1]
            note = chart[timestamps[i]]
            prev_note = chart[timestamps[i-1]]

            # Mode-agnostic jack detection
            if time_diff <= self.jack_threshold:
                for j in range(num_panels):
                    if note[j] == '1' and prev_note[j] == '1':
                        jacks += 1

            # Mode-agnostic crossover detection
            # A crossover happens when the center of mass of the feet crosses the midline
            current_note_panels = [j for j, val in enumerate(note) if val == '1']
            if not current_note_panels:
                current_note_com = last_note_com
            else:
                current_note_com = np.mean(current_note_panels)

            # Midline of the pad
            midline = (num_panels - 1) / 2.0

            # Check if the center of mass has crossed the midline
            if (last_note_com > midline and current_note_com < midline) or \
               (last_note_com < midline and current_note_com > midline):
                crossovers += 1

            last_note_com = current_note_com

        total_notes = len(timestamps)
        jack_percentage = (jacks / total_notes) * 100 if total_notes > 0 else 0
        crossover_percentage = (crossovers / total_notes) * 100 if total_notes > 0 else 0

        return {
            'jack_percentage': jack_percentage,
            'crossover_percentage': crossover_percentage
        }
