"""
This script contains functions to load, modify, work with the participants.tsv file
using pandas dataframes.
"""

import os
import pandas as pd


def get_participants_df(path_bids, participants_filename=None):
    if participants_filename is None:
        return pd.read_csv(path_bids / "participants.tsv", sep="\t")
    else:
        return pd.read_csv(path_bids / participants_filename, sep="\t")


def subset_participants_df(participants_df, subset_participants):
    """
    Returns the participants table in dataframe format, indexing only the
    participants in the subset iterable subset_participants"""
    subset_participants = [  # Format containing all "sub-" before the id
        f"sub-{sub}" if "sub-" not in sub else sub for sub in subset_participants
    ]
    idx_subset = participants_df["participant_id"].isin(subset_participants)
    return participants_df[idx_subset]


def add_metadata_at_session_to_participants_df(
    path_bids, participants_df, metadata_key, which_session=0
):
    """
    Adds a column with additional session-dependent metadata (such as diagnosis at a
    given visit) to the participants_df dataframe. Additionally adds an extra
    "{metadata_key}_ses" column that contains the information from which session the
    metadata has been collected.

    WARNING: THE BIDS DATASET MUST CONTAIN A "sub-XXX_sessions.tsv" table file for
    each subject, where the metadata collected at each session must be stored.

    :param path_bids: path-like object pointing to the BIDS dataset.
    :param participants_df: pd.DataFrame object from reading a participants.tsv
    :param metadata_key: str with the key of the target metadata info.
    :param which_session: Index of what session to choose from, when sorted alphanum.
    :return:
    """
    # Add additional columns that we will fill up
    participants_df[[metadata_key, f"{metadata_key}_ses"]] = None
    for sub in participants_df["participant_id"]:  # Iterate over participants
        # Each participant must have N "ses-XXX" directories
        sessions = [ses for ses in os.listdir(path_bids / sub) if "ses" in ses]
        sorted_sessions = sorted(sessions)
        target_ses = sorted_sessions[which_session]

        # As well as a sub-xxx_sessions.tsv file containing info for each ses
        df_sessions = pd.read_csv(path_bids / sub / f"{sub}_sessions.tsv", sep="\t")
        ses_idx = df_sessions["session_id"] == target_ses

        # Add the information from the given participant to the participants_df
        sub_idx = participants_df["participant_id"] == sub
        participants_df.loc[sub_idx, metadata_key] = df_sessions.loc[
            ses_idx, metadata_key
        ].to_numpy()[0]
        participants_df.loc[sub_idx, f"{metadata_key}_ses"] = target_ses
    return participants_df
