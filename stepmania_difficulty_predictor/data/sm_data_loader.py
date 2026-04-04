import os
import simfile
from typing import List
import sys

def load_sm_files_from_directory(directory: str) -> List[simfile.Simfile]:
    """
    Recursively finds and parses StepMania simfiles in a directory.

    Preference order:
    - If a song folder contains an `.ssc`, use that file.
    - Otherwise, fall back to `.sm`.

    Args:
        directory: The path to the directory to search.

    Returns:
        A list of parsed simfile objects.
    """
    stepfiles_by_dir = {}
    for root, _, files in os.walk(directory):
        for file in files:
            lower = file.lower()
            filepath = os.path.join(root, file)
            if lower.endswith('.sm'):
                stepfiles_by_dir.setdefault(root, filepath)
            elif lower.endswith('.ssc'):
                stepfiles_by_dir[root] = filepath

    sm_files = []
    for filepath in sorted(stepfiles_by_dir.values()):
        try:
            sm_files.append(simfile.open(filepath, strict=False))
        except Exception as e:
            print(f"Error parsing {filepath}: {e}", file=sys.stderr)
    return sm_files
