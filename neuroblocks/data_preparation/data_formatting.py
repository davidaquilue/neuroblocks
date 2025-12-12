"""
Module with functions for formatting data in our desired formats.
"""

import pandas as pd

def loader_data_to_long_df(loader_data, value_name="value"):
    """
    Data Loader data is stored in dictionary-like structure following:

        {subject: {session: {modality: {data, path} } } } }

    We transform the data (assumed to be (n_rois, ) in this case and transform it
    into a long-format DataFrame with columns:
        ['participant_id', 'ses', 'ses_i', 'modality', 'roi', 'value_name'].

    :param loader_data: output of Loader.data after running get_parcellated_data()
    :param value_name: name of the value column (such as CL, suvr, ...). Default
        "value" in loader_data
    :return: pandas.DataFrame
    """
    rows_df = []
    for sub in loader_data.keys():
        sessions = list(loader_data[sub].keys())
        sessions.sort()
        for ses_i, ses in enumerate(sessions):
            sub_ses_dict = loader_data[sub][ses]
            if "pet" in sub_ses_dict:  # PET is special case as it may have
                # additional values.
                # TODO This approach does not consider single-subject with multiple
                #  PET parcellated data loaded (say both AB and Tau)
                if "18FAV45" in sub_ses_dict["pet"]:
                    parc_data = sub_ses_dict["pet"]["18FAV45"]["data"]
                    modality = "pet-18FAV45"
                elif "18FFBB" in sub_ses_dict["pet"]:
                    parc_data = sub_ses_dict["pet"]["18FFBB"]["data"]
                    modality = "pet-18FFBB"
                elif "18FAV1451" in sub_ses_dict["pet"]:
                    parc_data = sub_ses_dict["pet"]["18FAV1451"]["data"]
                    modality = "pet-18FAV1451"
                else:
                    continue
            else:
                modality = list(sub_ses_dict.keys())[0]
                parc_data = sub_ses_dict[modality]["data"]

            for i, value in enumerate(parc_data):
                rows_df.append(
                    {"participant_id": f"sub-{sub}",
                     "ses": ses,
                     "ses_i": ses_i,
                     "modality": modality,
                     "roi": i,
                     value_name: value}
                )
    return pd.DataFrame(rows_df)
