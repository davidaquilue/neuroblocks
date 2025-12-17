"""
Functions to load parcellated data from ADNI.
"""

import re
import warnings
from .loader import FlexibleBIDSLoader, load_numpy_data_to_leaves
from ..feature_extraction.bold_features import compute_fc


class PETParcellatedLoader(FlexibleBIDSLoader):
    """
    Initialize the PETParcellatedLoader.

    :param dir_to_load (str): Path to the directory containing BIDS-structured PET data.
    :param parcellation_str (str): A string identifier for the parcellation scheme
        used in preprocessing (e.g., 'AAL', 'Schaefer').
    :param in_centiloid (bool): Whether the PET values are expressed in Centiloid units.
        If True, expects filenames with "_CL.npy" suffix.
    :param subset_participants (list or None): Optional list of participant IDs to limit
        the data loading to a subset.
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


class BOLDParcellatedLoader(FlexibleBIDSLoader):
    """
    fMRIParcellatedLoader to load parcellated resting state fMRI BOLD signals.

    :param dir_to_load (str): Path to the directory containing BIDS-structured
    preprocessed and denoised BOLD data.
    :param parcellation_str (str): A string identifier for the parcellation scheme
        used in preprocessing (e.g., 'AAL', 'Schaefer').
    :param subset_participants (list or None): Optional list of participant IDs to limit
        the data loading to a subset.
    :param prepro_prefix (str): string containing the prefix of the preprocessing
        steps undertaken by conn, defaults to "dswar" (coregistered, aligned,
        warped, smoothed, denoised).
    """

    def __init__(
        self,
        dir_to_load,
        parcellation_str,
        subset_participants=None,
    ):
        super().__init__(dir_to_load, subset_participants=subset_participants)
        self.parcellation_str = parcellation_str
        bold_re = r"sub-(?P<sub>[^_]+)_ses-(?P<ses>[^_]+)_task-rest_(?P<modality>bold)_"
        self.pattern = re.compile(bold_re + self.parcellation_str + r".npy")


    def get_parcellated_data(self):
        # Performs a scan of the data, and then goes over all the obtained paths and
        # loads the parcellated data in numpy format.
        self._scan()
        load_numpy_data_to_leaves(self.data)
        return self.data

    def get_parcellated_data_in_simple_dict(self):
        """
        Gets data and returns it in a dictionary of the type:
        {"sub_id": {"session": np.array(parcellated_data)}, ... }

        :return dict: Nested dictionary {subject_id: {tracer: np.ndarray, ...}, ...}
        """
        # Ensure data is scanned and loaded
        self.get_parcellated_data()

        simple_dict = {}
        for subject, sessions in self.data.items():  # Iterate over sessions and subs
            # Initialize per-subject dict
            simple_dict[subject] = {}

            # Directly load the available sessions
            for session in sessions:
                bold_data = sessions.get(session, {}).get("bold", {})
                if not isinstance(bold_data.get("data"), type(None)):
                    simple_dict[subject][session] = bold_data.get("data")
                else:  # Do not add the session to the subject
                    pass
            if len(simple_dict[subject]) == 0:  # Remove subjects without data
                del simple_dict[subject]

        return simple_dict

    def get_fc_in_simple_dict(self, fisher_z=True):
        """
        Gets data and returns it in a dictionary of the type:
        {"sub_id": {"session": np.array(parcellated_data)}, ... }

        :return dict: Nested dictionary {subject_id: {tracer: np.ndarray, ...}, ...}
        """
        # Get simple dict with BOLD signals
        bold_dict = self.get_parcellated_data_in_simple_dict()
        # Iterate and obtain FC through Pearson correlation (z-transform as well, opt)
        simple_dict_fc = {}
        for subject, sessions in bold_dict.items():  # Iterate over sessions and subs
            simple_dict_fc[subject] = {}
            for session in sessions:
                simple_dict_fc[subject][session] = compute_fc(
                    bold_dict[subject][session], fisher_z=fisher_z
                )
        return simple_dict_fc


class GMVParcellatedLoader(FlexibleBIDSLoader):
    """
    GMVParcellatedLoader to load parcellated Gray Matter Volumes.

    Slightly specific DataLoader as SHOOT templates can be generated for different
    subgroups. For instance, in ADNI, we can generate the following sets of templates:
    - "all_subjects": only one template taking into account all the subjects in the
     dataset
    - "cognormal" / "impaired": one template for CN, another template for MCI/AD
    - group-templates: for instance one template for CN, another template for MCI,
     another for AD. Or even stratifying them by Abeta positivity.

     Therefore, we pass an iterable of strs with the names of the templates (each
     will have a subdirectory in the templates_root_dir), from which we will look for
     the parcellated data.

    :param template_names (iterable of str): template_names for which we will look for
    parcellated data.
    :param templates_root_dir (Path): Path to the directory containing the directories .
    :param parcellation_str (str): A string identifier for the parcellation scheme
        used in preprocessing (e.g., 'Schaefer400', 'Glasser').
    :param subset_participants (list or None): Optional list of participant IDs to limit
        the data loading to a subset.
    :param load_atrophy_rates (bool): Whether to load the atrophy rates (Gray
    Matter Volume Annualized Percentage Change, GMvolAPC). Only one per individual,
    in ses-avg.
    """

    def __init__(
        self,
        template_names,
        templates_root_dir,
        parcellation_str,
        subset_participants=None,
        load_atrophy_rates=False,
    ):
        # We initialize with templates root_dir but will change the dir_to_load
        # attribute dynamically when scanning the data.
        super().__init__(templates_root_dir, subset_participants=subset_participants)
        self.parcellation_str = parcellation_str
        if load_atrophy_rates:
            gmv_re = r"sub-(?P<sub>[^_]+)_ses-(?P<ses>[^_]+)_(?P<modality>GMvolAPC)_"
        else:
            gmv_re = r"sub-(?P<sub>[^_]+)_ses-(?P<ses>[^_]+)_(?P<modality>GMvol)_"
        self.pattern = re.compile(gmv_re + self.parcellation_str + r".npy")
        self.templates_root_dir = templates_root_dir
        self.template_names = template_names

    def get_parcellated_data(self):
        # Performs a scan of the data, and then goes over all the obtained paths and
        # loads the parcellated data in numpy format.
        for template in self.template_names:
            self.dir_to_load = self.templates_root_dir / template
            self._scan()
            load_numpy_data_to_leaves(self.data)
        return self.data

    def get_parcellated_data_in_simple_dict(self):
        """
        Gets data and returns it in a dictionary of the type:
        {"sub_id": {"session": np.array(parcellated_data)}, ... }

        :return dict: Nested dictionary {subject_id: {session_id: np.ndarray, ...}, ...}
        """
        # Ensure data is scanned and loaded
        self.get_parcellated_data()

        simple_dict = {}
        for subject, sessions in self.data.items():  # Iterate over sessions and subs
            # Initialize per-subject dict
            simple_dict[subject] = {}

            # Directly load the available sessions
            for session in sessions:
                gmvol_data = sessions.get(session, {}).get("GMvol", {})
                if not isinstance(gmvol_data.get("data"), type(None)):
                    simple_dict[subject][session] = gmvol_data.get("data")
                else:  # Do not add the session to the subject
                    pass
            if len(simple_dict[subject]) == 0:  # Remove subjects without data
                del simple_dict[subject]

        return simple_dict
