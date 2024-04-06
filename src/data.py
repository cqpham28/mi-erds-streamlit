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


class Flex2023_moabb_st(BaseDataset):
    """Motor Imagery moabb dataset 
    Adapt to streamlit upload individual edf files"""

    def __init__(self):
        super().__init__(
            subjects=list(range(12, 20)),
            sessions_per_subject=1,
            events=dict(right_hand=1, left_hand=2, right_foot=3, left_foot=4),
            # events=dict(right_hand=1, left_hand=2, right_foot=3, left_foot=4,
            #     right_hand_kines=11, left_hand_kines=12, 
            #     right_foot_kines=13, left_foot_kines=14),
            code="Flex2023",
            interval=[4, 8], # events at 4s
            paradigm="imagery",
            doi="",
        )
        self.runs = "run1"
        

    def _flow(self, raw0, stim):
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
        # raw.set_eeg_reference(ref_channels="average")
        
        return raw


    def _get_single_subject_data(self, subject):
        """Return data for a single subject."""
    
        path_edf = self.data_path(subject)
        raw0 = mne.io.read_raw_edf(path_edf, preload=False)
        stim = raw0.get_data(picks=["MarkerValueInt"], units='uV')[0]
        raw = self._flow(raw0, stim)

        return {"0": {"0": raw}}



    def data_path(self, subject, path=None, force_update=False, update_path=None, verbose=None):
        ## ADAPT STREAMLIT 
        return st.session_state.path_edf[f"{subject}-{self.runs}"]




