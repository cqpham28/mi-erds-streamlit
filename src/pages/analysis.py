"""
"""
import os
import streamlit as st
from src.ml_new import run_ml, plot_ml
import src.utils as utils
import streamlit as st




def write():
    """Used to write the page in the app.py file"""

    st.header("Analysis")
    st.markdown(
        """
        :red[This is the processing pipeline, using different ML BASELINE models:]
        - :red[For protocol 4c, we used data 0-2s after MI onset.] 
        - :red[For protocol 8c, we used 0-2s for MI data, we used 3-5s for Rest/NoRest data.]  
        
        """
    )

    ##--------GET FILE---------##
    protocol, subject, session, run, path_file = utils.select_box_to_file()

    ##
    button_clear = st.sidebar.button(":blue[Clear Cache]")
    if button_clear:
        st.cache_data.clear()

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
                list_model_name = ["8c_rest", "8c_mi"]
            elif "4c" in protocol:
                list_model_name = ["4c_2class_hand", "4c_2class_foot", "4c_all"]
                
            # run benchmark ML
            for model_name in list_model_name:
                
                with st.expander(f":blue[{model_name}]", expanded=False):
                    df = run_ml(protocol, subject, session, run, model_name)
                    col1, col2 = st.columns([1,3])
                    with col1: 
                        st.write(df)
                    with col2: 
                        img = plot_ml(df)
                        st.image(img)