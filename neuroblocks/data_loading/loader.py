"""
Contains a class for loading data from a BIDS-formatted dataset.

It allows for flexibility in loading different types of data using regular
expressions both from the root BIDS or derivatives folder.
"""

import re

import warnings
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
    "any": r"sub-(?P<sub>[^_]+)_ses-(?P<ses>[^_]+).*_(?P<modality>[a-zA-Z0-9]+).*\.nii(\.gz)?",
}


def load_numpy_data_to_leaves(d):
    """
    Recursively traverse a nested dictionary and, at each leaf-level dictionary,
    add a key 'data' with an empty dictionary as its value (or some other content).
    """
    for _, value in d.items():
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

    def __init__(self, dir_to_load, pattern="any", subset_participants=None):
        """
        Initialize the loader.

        Args:
            dir_to_load (str or Path): Path to the BIDS dataset root.
            pattern (str): Either a regex string or a key from PRESETS.
            subset_participants: iterable containing a subset of the participants
            from the dataset to gather data from them. Default None.
        """
        self.PRESETS = PRESETS
        self.dir_to_load = Path(dir_to_load)
        self.data = defaultdict(lambda: defaultdict(dict))
        # We clean the participants if they are provided in BIDS format
        self.subset_participants = [
            sub.replace("sub-", "") for sub in subset_participants
        ]

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
            if self.subset_participants is not None:  # If we have passed
                if sub not in self.subset_participants:  # And sub not in list
                    continue  # skip this file
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
                            rows.append(
                                {
                                    "subject": subject,
                                    "session": session,
                                    "modality": modality,
                                    "tracer": tracer,
                                    "path": info.get("path"),
                                }
                            )
                    else:
                        # Normal modality
                        rows.append(
                            {
                                "subject": subject,
                                "session": session,
                                "modality": modality,
                                "tracer": None,
                                "path": val.get("path"),
                            }
                        )

        df_loader = pd.DataFrame(rows)
        return df_loader

    def average_data_over_participants(self, participant_ids, which_session=0):
        """
        Takes a list of participant_ids and averages the data in the "data" key,
        for the n-th session available in the dataset.

        WARNING: Requires data to have been loaded previously with a different method
        such as get_parcellated_data() in the PETParcellatedLoader subclass. The loaded
        data must follow the same structure as the _scan() method, but adding a
        "data" key with loaded data as value.

        :param participant_ids: list of str
        :param which_session: int
        :return: Returns a dictionary containing the averaged data over the
        participants. For "pet" modality, it returns keys a mean array per tracer, with
        "avg_{trc}" key, for other modalities, returns a single key with "avg".
        """
        # We will store in the data_lists the data. If the modality is PET and we
        # have more than one tracer, we will store it in the dictionary.
        # key will be either "avg" or "avg_trc"
        data_lists = {"avg": []}

        for subject, sessions in self.data.items():
            if subject not in participant_ids:
                # Iterate only over the subjects in participant_ids
                continue
            # Sort sessions alphanumerically and pick the one requested
            sorted_sessions = sorted(sessions.keys())
            if which_session >= len(sorted_sessions):
                warnings.warn(
                    f"Skipped {subject} - session index is out of range", stacklevel=2
                )
                continue
            session = sorted_sessions[which_session]
            # Get the modality of the data
            modality = list(sessions.get(session, {}).keys())[0]
            modality_val = sessions.get(session, {})[modality]
            if modality == "pet":
                for trc, trc_dict in modality_val.items():
                    if f"avg_{trc}" not in data_lists:
                        data_lists[f"avg_{trc}"] = []
                    data_lists[f"avg_{trc}"].append(trc_dict["data"])
            else:
                data_lists["avg"].append(modality_val["data"])

        if len(data_lists["avg"]) == 0:  # For PET cases with given trcrs.
            data_lists.pop("avg")  # Remove the "avg" key

        for key, data_list in data_lists.items():
            try:
                # Check whether it's numpy arrays
                data_arr = np.array(data_list)
                # Perform average
                data_lists[key] = np.mean(data_arr, axis=0)
            except TypeError:
                print(
                    f"Cannot convert data in {key} to numpy array. "
                    f"Implementation not available for data_type"
                )

        return data_lists
