import numpy as np

class StreamDetector:
    """
    Detects and quantifies streams of notes in a chart.
    """
    def __init__(self, stream_threshold=0.25):
        """
        Initializes the StreamDetector.

        Args:
            stream_threshold: The maximum time between notes to be considered a stream.
        """
        self.stream_threshold = stream_threshold

    def compute(self, chart: dict) -> dict:
        """
        Computes the stream features for a given chart.

        Args:
            chart: A dictionary representing the chart, with timestamps as keys.

        Returns:
            A dictionary containing the stream features.
        """
        timestamps = sorted(chart.keys())
        if len(timestamps) < 2:
            return {'stream_percentage': 0, 'max_stream_length': 0}

        stream_notes = 0
        max_stream_length = 0
        current_stream_length = 0

        for i in range(1, len(timestamps)):
            time_diff = timestamps[i] - timestamps[i-1]
            if time_diff <= self.stream_threshold:
                if current_stream_length == 0:
                    current_stream_length = 2
                else:
                    current_stream_length += 1
            else:
                if current_stream_length > 0:
                    stream_notes += current_stream_length
                max_stream_length = max(max_stream_length, current_stream_length)
                current_stream_length = 0

        if current_stream_length > 0:
            stream_notes += current_stream_length
        max_stream_length = max(max_stream_length, current_stream_length)

        total_notes = len(timestamps)
        stream_percentage = (stream_notes / total_notes) * 100 if total_notes > 0 else 0

        return {
            'stream_percentage': stream_percentage,
            'max_stream_length': max_stream_length
        }
