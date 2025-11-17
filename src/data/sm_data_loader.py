import os
import simfile
from typing import List

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
                    print(f"Error parsing {filepath}: {e}")
    return sm_files

if __name__ == '__main__':
    # This is a placeholder for where the data will be downloaded
    # For now, we'll assume a directory named 'data/raw' exists
    # and contains .sm files.
    raw_data_dir = "data/raw"
    if not os.path.exists(raw_data_dir):
        os.makedirs(raw_data_dir)
        print(f"Created directory {raw_data_dir}. Please add .sm files to this directory.")
    else:
        simfiles = load_sm_files_from_directory(raw_data_dir)
        print(f"Found {len(simfiles)} .sm files.")
        for sm in simfiles:
            print(f"- {sm.title}")
