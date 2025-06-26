"""
Contains a class for loading data from a BIDS-formatted dataset.

It allows for flexibility in loading different types of data using regular
expressions both from the root BIDS or derivatives folder.
"""

import re

import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

# Preset regex patterns for common BIDS modalities
PRESETS = {
    "t1w": r"sub-(?P<sub>[^_]+)_ses-(?P<ses>[^_]+)_(?P<modality>T1w).*\.nii(\.gz)?",
    "bold": r"sub-(?P<sub>[^_]+)_ses-(?P<ses>[^_]+)_(?P<modality>bold).*\.nii(\.gz)?",
    "dwi": r"sub-(?P<sub>[^_]+)_ses-(?P<ses>[^_]+)_(?P<modality>dwi).*\.nii(\.gz)?",
    "pet": r"sub-(?P<sub>[^_]+)_ses-(?P<ses>[^_]+)_(trc-(?P<tracer>[^-_]+)_)?"
           r"(?P<modality>pet).*\.nii(\.gz)?",
    "any": r"sub-(?P<sub>[^_]+)_ses-(?P<ses>[^_]+).*_(?P<modality>[a-zA-Z0-9]+).*\.nii(\.gz)?"
}


def load_numpy_data_to_leaves(d):
    """
    Recursively traverse a nested dictionary and, at each leaf-level dictionary,
    add a key 'data' with an empty dictionary as its value (or some other content).
    """
    for key, value in d.items():
        if isinstance(value, dict):
            # If value is a dict, go deeper
            if all(not isinstance(v, dict) for v in value.values()):
                # If the values of this dict are not dicts -> it's a leaf level
                value["data"] = np.load(value["path"])
            else:
                # Otherwise, keep traversing
                load_numpy_data_to_leaves(value)


class FlexibleBIDSLoader:
    """
    A flexible loader for BIDS-formatted datasets, supporting raw and derivative data,
    and customizable or preset regex patterns to extract subject, session, and modality info.
    """

    def __init__(self, dir_to_load, pattern="any"):
        """
        Initialize the loader.

        Args:
            dir_to_load (str or Path): Path to the BIDS dataset root.
            pattern (str): Either a regex string or a key from PRESETS.
        """
        self.PRESETS = PRESETS
        self.dir_to_load = Path(dir_to_load)
        self.data = defaultdict(lambda: defaultdict(dict))

        # Compile pattern
        if pattern in self.PRESETS:
            self.pattern = re.compile(self.PRESETS[pattern])
        else:
            self.pattern = re.compile(pattern)

    def _scan(self):
        """
        Scan a directory recursively and extract matching file info.
        """
        for file in self.dir_to_load.rglob("*"):
            match = self.pattern.search(file.name)
            if not match:
                continue

            info = match.groupdict()
            # Pseudocode inside your _scan method after matching:

            sub = info.get("sub", "unknown")
            ses = info.get("ses", "unknown")
            modality = info.get("modality", "unknown")
            tracer = info.get("tracer")  # might be None if missing

            if modality == "pet" and tracer:
                # store with tracer as additional key
                if "pet" not in self.data[sub][ses]:
                    self.data[sub][ses]["pet"] = {}
                self.data[sub][ses]["pet"][tracer] = {"path": str(file)}
            else:
                # store as usual
                self.data[sub][ses][modality] = {"path": str(file)}

    def get_paths(self):
        """
        Retrieve the loaded data structure.

        Returns:
            dict: Nested dictionary of the form data[subject][session][modality] -> info
        """
        self._scan()
        return self.data


    def nested_dict_to_dataframe(self):
        """
        Convert nested data dict of the form:
          data[subject][session][modality or modality][tracer?] = info_dict
        into a pandas DataFrame with columns:
          ['subject', 'session', 'modality', 'tracer', 'path', 'source']

        Handles optional tracer level for PET scans.
        """
        rows = []
        if len(self.data) == 0:
            self._scan()

        for subject, sessions in self.data.items():
            for session, modalities in sessions.items():
                for modality, val in modalities.items():
                    # Check if PET modality has nested tracer dict
                    if modality == "pet" and isinstance(val, dict):
                        for tracer, info in val.items():
                            rows.append({
                                "subject": subject,
                                "session": session,
                                "modality": modality,
                                "tracer": tracer,
                                "path": info.get("path"),
                            })
                    else:
                        # Normal modality
                        rows.append({
                            "subject": subject,
                            "session": session,
                            "modality": modality,
                            "tracer": None,
                            "path": val.get("path"),
                        })

        df = pd.DataFrame(rows)
        return df
