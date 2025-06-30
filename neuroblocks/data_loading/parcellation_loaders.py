"""
Functions to load parcellated data from ADNI.
"""

import re
import warnings
from .loader import FlexibleBIDSLoader, load_numpy_data_to_leaves


class PETParcellatedLoader(FlexibleBIDSLoader):
    """
    Class to load Parcellated PET data as output from our pre-processing pipeline.
    Inherits from our FlexibleBIDSLoader, which requires a specific regex pattern to
    be implemented which will be used to sift through the different files and load
    the relevant ones.
    """

    def __init__(
        self,
        dir_to_load,
        parcellation_str,
        in_centiloid=False,
        subset_participants=None,
    ):
        super().__init__(dir_to_load, subset_participants=subset_participants)
        self.parcellation_str = parcellation_str
        self.in_centiloid = in_centiloid
        pet_re = r"sub-(?P<sub>[^_]+)_ses-(?P<ses>[^_]+)(?:_trc-(?P<tracer>[^_]+))?_(?P<modality>pet)_"
        if in_centiloid:
            self.pattern = re.compile(pet_re + self.parcellation_str + r"_CL\.npy")
        else:
            self.pattern = re.compile(pet_re + self.parcellation_str + r"\.npy")

    def get_parcellated_data(self):
        # Performs a scan of the data, and then goes over all the obtained paths and
        # loads the parcellated data in numpy format.
        self._scan()
        load_numpy_data_to_leaves(self.data)
        return self.data

    def get_parcellated_data_in_simple_dict(self, trcrs="all", which_session=0):
        """
        Gets data and returns it in a dictionary of the type:
        {"sub_id": {"trc": np.array(parcellated_data)}, ... }

        :param trcrs: list of strings with the tracer string identifiers
        :param which_session: idx of which session from the available sessions to
        take. They will be ordered alphanumerically.
        :return dict: Nested dictionary {subject_id: {tracer: np.ndarray, ...}, ...}

        """
        # Ensure data is scanned and loaded
        self.get_parcellated_data()

        simple_dict = {}

        for subject, sessions in self.data.items():
            # Sort sessions alphanumerically and pick the one requested
            sorted_sessions = sorted(sessions.keys())
            if which_session >= len(sorted_sessions):
                warnings.warn(
                    f"Skipped {subject} - session index is out of range", stacklevel=2
                )
                continue

            session = sorted_sessions[which_session]
            pet_data = sessions.get(session, {}).get("pet", {})

            if not isinstance(pet_data, dict):
                continue  # Skip if PET data structure is not as expected

            # Initialize per-subject dict
            simple_dict[subject] = {}

            for tracer, info in pet_data.items():
                if trcrs != "all" and tracer not in trcrs:
                    continue
                data = info.get("data")
                if data is not None:
                    simple_dict[subject][tracer] = data

            # If no tracers matched, remove the subject key
            if not simple_dict[subject]:
                del simple_dict[subject]

        return simple_dict
