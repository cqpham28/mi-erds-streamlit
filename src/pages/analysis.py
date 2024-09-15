"""
"""
import os
import streamlit as st
from src.ml_new import run_ml
import src.utils as utils
import streamlit as st




def write():
    """Used to write the page in the app.py file"""

    st.header("Analysis")
    st.markdown(
        """:red[This is the processing pipeline, using different baseline models]"""
    )

    ##--------GET FILE---------##
    protocol, subject, session, run, path_file = utils.select_box_to_file()


    ##------PROCESSING---------##
    button_analyze = st.button(":red[Analyze]")
    if button_analyze:

        # 1. gget data from S3
        key = f"F{subject}_{protocol}_{session}_{run}"
        path_save = f"refs/{key}.edf"
        utils.get_edf_s3(path_file=path_file, 
                   path_save=path_save)
        st.session_state.path_edf[key] = path_save

        # 2. RUN BASELINE MODEL
        with st.spinner(":blue[RUNNING BASELINE MODEL...]"):
            # choose model
            if "8c" in protocol:
                list_model_name = ["8c_hand", "8c_mi", "8c_rest"]
            elif "4c" in protocol:
                list_model_name = ["4c_rest", "4c_2class_hand", "4c_2class_foot",
                                   "4c_2class_handfoot", "4c_all"]
                
            # run benchmark ML
            for model_name in list_model_name:
                with st.expander(f":blue[{model_name}]", expanded=False):
                    df = run_ml(protocol, subject, session, run, model_name)
                    st.write(df)