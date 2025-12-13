import os
import simfile
from typing import List
import sys

def load_sm_files_from_directory(directory: str) -> List[simfile.Simfile]:
    """
    Recursively finds and parses all .sm files in a directory.

    Args:
        directory: The path to the directory to search.

    Returns:
        A list of parsed simfile objects.
    """
    sm_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".sm"):
                filepath = os.path.join(root, file)
                try:
                    sm_files.append(simfile.open(filepath))
                except Exception as e:
                    print(f"Error parsing {filepath}: {e}", file=sys.stderr)
    return sm_files
