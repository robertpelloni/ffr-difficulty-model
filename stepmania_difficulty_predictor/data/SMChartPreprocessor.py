import sys
import numpy as np
import simfile
from simfile.timing import TimingData
from simfile.timing.engine import TimingEngine
from simfile.notes import NoteData, NoteType

class SMChartPreprocessor:
    """
    Preprocesses a simfile object to a dictionary in the following format:
    {
        'name': name of stepfile (str),
        'difficulty': difficulty of the chart (str),
        'meter': meter of the chart (int),
        'chart': {
            timestamps (float): binary step encodings (str)
        }
    }
    """

    def __init__(self, decimals=3):
        self.decimals = decimals
        self.mappings = {
            '1': 1000,  # Left
            '2': 100,   # Down
            '3': 10,    # Up
            '4': 1,     # Right
            'M': 1000,  # Mine (maps to Left for now)
            # Other note types like holds, rolls, etc., are ignored for now
        }

    def preprocess(self, sm_file: simfile.Simfile):
        """
        Processes each chart in a simfile.
        """
        preprocessed_charts = []
        if not hasattr(sm_file, 'charts') or not sm_file.charts:
            return preprocessed_charts

        timing_data = TimingData(sm_file)
        timing_engine = TimingEngine(timing_data)

        for chart in sm_file.charts:
            if not chart or chart.stepstype != 'dance-single':
                continue

            note_data = NoteData(chart)
            timed_notes = []
            for note in note_data:
                if note.note_type == NoteType.TAP:
                    try:
                        time = timing_engine.time_at(note.beat)
                        encoding = self._encode_note(note.column)
                        if encoding != 0:
                            timed_notes.append((time, encoding))
                    except (ValueError, KeyError):
                        # Skip notes with invalid columns
                        continue

            if not timed_notes:
                continue

            # Create the chart dictionary
            chart_dict = {
                np.round(time, self.decimals): str(encoding).zfill(4)
                for time, encoding in timed_notes
            }

            difficulty = getattr(chart, 'difficulty', 'Unknown')
            if difficulty.isdigit():
                difficulty_map = {'1': 'Beginner', '2': 'Easy', '3': 'Medium', '4': 'Hard', '5': 'Challenge'}
                difficulty = difficulty_map.get(difficulty, 'Unknown')

            meter = 0
            meter_str = getattr(chart, 'meter', '0')
            if meter_str and meter_str.isdigit():
                meter = int(meter_str)

            preprocessed_charts.append({
                'name': getattr(sm_file, 'title', 'Unknown'),
                'difficulty': difficulty,
                'meter': meter,
                'chart': chart_dict,
            })

        return preprocessed_charts

    def _encode_note(self, column: int) -> int:
        """
        Encodes a note column into an integer representation.
        """
        # Based on 4-key dance single
        note_map = {
            0: 1000, # Left
            1: 100,  # Down
            2: 10,   # Up
            3: 1,    # Right
        }
        return note_map.get(column, 0)
