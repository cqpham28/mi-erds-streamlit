"""

"""
import streamlit as st
import numpy as np
import mne
from moabb.datasets.base import BaseDataset
from src.config import *


#=========================#
class Flex2023_moabb_st(BaseDataset):
    """Motor Imagery moabb dataset 
    Adapt to streamlit implementation"""

    def __init__(self, protocol:str="8c", session:str="ss1", run:str="run1"):

        if "4c" in protocol:
            events = dict(right_hand=1, left_hand=2, right_foot=3, left_foot=4)
        elif "8c" in protocol:
            events = dict(right_hand=1, left_hand=2, right_foot=3, left_foot=4,
                        right_hand_r=5, left_hand_r=6, right_foot_r=7, left_foot_r=8)
            
        super().__init__(
            subjects=LIST_SUBJECTS,
            sessions_per_subject=1,
            events=events,
            code="Flex2023",
            interval=[4, 8], # events at 4s
            paradigm="imagery",
            doi="",
        )
        # self.runs = "run1"
        # self.key = key
        self.protocol = protocol
        self.session = session
        self.run = run
        

    def _flow(self, raw0, stim):
        """Single flow of raw processing"""

        ## get eeg (32,N)
        data = raw0.get_data(picks=EEG_CH_NAMES)
        
        # stack eeg (32,N) with stim (1,N) => (32, N)
        data = np.vstack([data, stim.reshape(1,-1)])

        ch_types = ["eeg"]*32 + ["stim"]
        ch_names = EEG_CH_NAMES + ["Stim"]
        info = mne.create_info(ch_names=ch_names, 
                            ch_types=ch_types, 
                            sfreq=SAMPLING_RATE)
        raw = mne.io.RawArray(data=data, 
                            info=info, 
                            verbose=False)
        montage = mne.channels.make_standard_montage("standard_1020")
        raw.set_montage(montage)
        
        ## filter
        raw.filter(l_freq=FILTER_LOWCUT, h_freq=None, method='iir') \
            .notch_filter(freqs=[FILTER_NOTCH])
        
        return raw


    def _get_single_subject_data(self, subject:int):
        """Return data for a single subject."""
    
        path_edf = self.data_path(subject)
        raw0 = mne.io.read_raw_edf(path_edf, preload=False)
        stim = raw0.get_data(picks=["MarkerValueInt"], units='uV')[0]
        raw = self._flow(raw0, stim)

        return {"0": {"0": raw}}



    def data_path(
        self, 
        subject, 
        path=None, 
        force_update=False, 
        update_path=None, 
        verbose=None
        ):
        ## ADAPT STREAMLIT 
        key = f"F{subject}_{self.protocol}_{self.session}_{self.run}"
        return st.session_state.path_edf[key]




