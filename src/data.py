"""

"""
import streamlit as st
import os
import numpy as np
import pandas as pd
import mne
from mne.channels import make_standard_montage
from mne.io import read_raw_edf
from moabb.datasets.base import BaseDataset


class Flex2023(BaseDataset):
    """Motor Imagery moabb dataset 
    Adapt to streamlit upload edf files"""

    def __init__(self):
        super().__init__(
            subjects=list(range(15)),
            sessions_per_subject=1,
            events=dict(right_hand=1, left_hand=2, right_foot=3, left_foot=4),
            code="Flex2023",
            interval=[4, 8], # events at 4s
            paradigm="imagery",
            doi="",
        )

    def _get_single_subject_data(self, subject):
        """Return data for a single subject."""
    
        ## read raw
        path_edf = self.data_path(subject)
        raw0 = st_read_edf(path_edf)

        ## stim events
        # stim = raw0.get_data(picks=["MarkerValueInt"], units='uV')[0]
        stim = fix_stim(raw0)

        # fmt: off
        eeg_ch_names = [
            'Cz', 'Fz', 'Fp1', 'F7', 'F3', 
            'FC1', 'C3', 'FC5', 'FT9', 'T7', 
            'CP5', 'CP1', 'P3', 'P7', 'PO9', 
            'O1', 'Pz', 'Oz', 'O2', 'PO10', 
            'P8', 'P4', 'CP2', 'CP6', 'T8', 
            'FT10', 'FC6', 'C4', 'FC2', 'F4', 
            'F8', 'Fp2'
        ]

        ## get eeg (32,N)
        data = raw0.get_data(picks=eeg_ch_names)
        # stack eeg (32,N) with stim (1,N) => (32, N)
        data = np.vstack([data, stim.reshape(1,-1)])

        ch_types = ["eeg"]*32 + ["stim"]
        ch_names = eeg_ch_names + ["Stim"]
        info = mne.create_info(ch_names=ch_names, ch_types=ch_types, sfreq=128)
        raw = mne.io.RawArray(data=data, info=info, verbose=False)
        montage = make_standard_montage("standard_1020")
        raw.set_montage(montage)

        return {"0": {"0": raw}}


    def data_path(self, subject, path=None, force_update=False, update_path=None, verbose=None):
        return st.session_state.path_edf[subject]

    



def fix_stim(edf_raw):
    """ fix stim for old procedure """

    ## get markers (biocalib+kines+ mi). This is the event of trial.
    markerIndex = edf_raw.get_data(picks=["MarkerIndex"], units='uV')[0]
    markers = np.where(markerIndex != 0)[0] # (320,)

    ## stim fix (because we assign value0)
    markerValueInt = edf_raw.get_data(picks=["MarkerValueInt"], units='uV')[0]
    stim = np.zeros_like(markerValueInt)
    for i, value in enumerate(markers):
        # 120 biocalib, offset = 20
        if 0 <= i < 120:
            offset = 20
            stim[value] = markerValueInt[value] + offset
        
        # 20 kines, offset = 10
        elif 120 <= i < 140:
            offset = 10
            stim[value] = markerValueInt[value] + offset
        
        # 140 MI, 1,2,3,4
        else:
            if markerValueInt[value] == 0:
                stim[value] = 1
            else:
                stim[value] = markerValueInt[value] + 1

    # ## plot check
    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots(3, 1, sharex=True) 
    # ax[0].plot(markerIndex)
    # ax[0].set_title("markerIndex")
    # ax[1].plot(markerValueInt, label="markerValueInt")
    # ax[1].set_title("markerValueInt")
    # ax[2].plot(stim, label="markerValueInt_fixed")
    # ax[2].set_title("markerValueInt_fixed")
    # plt.show()

    return stim




@st.cache_data
def st_read_edf(path):
    """read edf file from the working directory path"""
    return mne.io.read_raw_edf(path, preload=False)

