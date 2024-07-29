"""
conda create -n st_bci python=3.9
pip install -r requirements.txt
"""

import os
import streamlit as st
import src.utils as utils
from src.pages import home, analysis, feedback, benchmark

st.set_page_config(
    page_title="app",
    layout="wide",
)

#---------------#
# if not check_password():
#     st.stop()

#--------------------------#
PAGES = {
    "Home": home,
    "Offline Analysis": analysis,
    # "Feedback": feedback,
    # "Benchmark": benchmark,
}

##-------PARAMETER------##
if "s3_df" not in st.session_state: 
    st.session_state.s3_df = []

if "s3_img" not in st.session_state: 
    st.session_state.s3_img = []

if "current_run" not in st.session_state: 
    st.session_state.current_run = 0

if "current_file" not in st.session_state: 
    st.session_state.current_file = ""

if "path_edf" not in st.session_state: 
    st.session_state.path_edf = {}

if "data_run" not in st.session_state: 
    st.session_state.data_run = {
        "hand": {
            "event_ids": dict(left_hand=1, 
                            right_hand=2),
            },

        "foot": {
            "event_ids": dict(left_foot=4, 
                            right_foot=3),
            },
    }




def main():

    selection = st.sidebar.radio("Select pages", 
                    list(PAGES.keys()))
    os.makedirs("refs", exist_ok=True)
    
    page = PAGES[selection]
    with st.spinner(f"Loading {selection} ..."):
        page.write()


if __name__ == "__main__":
    main()