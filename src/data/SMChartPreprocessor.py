import numpy as np
import simfile
from simfile.timing import TimingData
from simfile.notes import NoteData

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
        timing_data = TimingData(sm_file)

        for chart in sm_file.charts:
            # We are only considering single player dance charts for now
            if chart.stepstype != 'dance-single':
                continue

            note_data = NoteData(chart)
            timed_notes = []
            for note in note_data:
                # Only consider tap notes for now
                if note.note_type == 'Tap':
                    time = timing_data.time_at(note.beat)
                    encoding = self._encode_note(note.column)
                    if encoding != 0:
                        timed_notes.append((time, encoding))

            if not timed_notes:
                continue

            # Create the chart dictionary
            chart_dict = {
                np.round(time, self.decimals): str(encoding).zfill(4)
                for time, encoding in timed_notes
            }

            difficulty = chart.difficulty
            if difficulty.isdigit():
                difficulty_map = {'1': 'Beginner', '2': 'Easy', '3': 'Medium', '4': 'Hard', '5': 'Challenge'}
                difficulty = difficulty_map.get(difficulty, 'Unknown')

            preprocessed_charts.append({
                'name': sm_file.title,
                'difficulty': difficulty,
                'meter': int(chart.meter),
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

if __name__ == '__main__':
    # Example Usage:
    # This part is for testing and demonstration.
    # It requires a sample .sm file.

    # Create a dummy sm file for testing
    sm_content = """
#TITLE:Test Song;
#ARTIST:Test Artist;
#BPMS:0=120;
#NOTES:
     dance-single:
     Beginner:
     1:
     :
1000
0100
0010
0001
;
"""
    with open("test.sm", "w") as f:
        f.write(sm_content)

    sm = simfile.open("test.sm")
    preprocessor = SMChartPreprocessor()
    processed_data = preprocessor.preprocess(sm)
    print(processed_data)

    import os
    os.remove("test.sm")
