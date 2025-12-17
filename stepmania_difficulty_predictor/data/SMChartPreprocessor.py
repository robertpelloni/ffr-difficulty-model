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
            if not chart or not chart.stepstype:
                continue

            note_data = NoteData(chart)

            # Group notes by timestamp
            notes_by_time = {}
            for note in note_data:
                if note.note_type == NoteType.TAP:
                    try:
                        time = timing_engine.time_at(note.beat)
                        time = np.round(time, self.decimals)
                        if time not in notes_by_time:
                            notes_by_time[time] = []
                        notes_by_time[time].append(note.column)
                    except (ValueError, KeyError):
                        continue

            if not notes_by_time:
                continue

            # Determine the number of panels
            if hasattr(chart, 'columns') and chart.columns:
                num_panels = len(chart.columns)
            else:
                # Fallback for older simfile versions or malformed charts
                mode_panels = {'dance-single': 4, 'dance-double': 8, 'pump-single': 5, 'pump-double': 10}
                num_panels = mode_panels.get(chart.stepstype, 0)

            if num_panels == 0:
                continue

            chart_dict = {
                time: self._encode_row(columns, num_panels)
                for time, columns in notes_by_time.items()
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
                'mode': chart.stepstype,
                'difficulty': difficulty,
                'meter': meter,
                'chart': chart_dict,
            })

        return preprocessed_charts

    def _encode_row(self, columns: list[int], num_panels: int) -> str:
        """
        Encodes a list of active columns for a given row/timestamp into a binary string.
        """
        encoding = ['0'] * num_panels
        for col in columns:
            if 0 <= col < num_panels:
                encoding[col] = '1'
        return "".join(encoding)
