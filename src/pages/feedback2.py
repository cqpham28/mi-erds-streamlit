import streamlit as st
import streamlit.components.v1 as components
import mne
from PIL import Image
import os
from src.plots import get_tfr, plot_curve, plot_heatmap
from src.ml_new import run_ml
import src.utils as utils
from src.config import *


# (09/2024) this is the NEW version
################ MAIN #################

def write():
    """Used to write the page in the app.py file"""

    st.header("MI Feedback")
    st.markdown(
        """:red[This is the baseline performance + ERDS feedback (for protocol 4c only)]"""
    )
    protocol = "4c"

    ##--------GET FILE---------##
    _, subject, session, run, path_file = utils.select_box_to_file(
        list_protocols=[protocol]
    )
    # 1. gget data from S3
    key = f"F{subject}_{protocol}_{session}_{run}"
    path_save = f"refs/{key}.edf"
    st.session_state.path_edf[key] = path_save
    
    ##
    st.session_state.current_subject = subject
    st.session_state.current_protocol = protocol
    st.session_state.current_session = session
    st.session_state.current_run = run

    ## [Level-2] Tab_ML & ERDS

    ##------PROCESSING---------##
    button = st.button(":red[Run]")
    if button:
        # get data from S3
        utils.get_edf_s3(path_file=path_file, path_save=path_save)
        
        tab1, tab2 = st.tabs(["ML", "ERDS"])

        # ML benchmark
        with tab1:
            st.markdown('''
                :blue[Check AUC-ROC/Accuracy of (LH-RH)/(LF-RF)/(4class) model]\n
                :blue[Apply 3 channels C3-Cz-C4. Using CSP(8-13Hz).]
                ''')

            with st.spinner("Running classifier..."):
                for model_name in ["4c_2class_hand", "4c_2class_foot"]:
                    with st.spinner(f"[{model_name}]"): 
                        df = run_ml(protocol, subject, session, run, model_name)
                        st.write(df)
                        # img = plot_ml_feedback(df)
                        # st.image(img)

            # ## Upload data 
            # with st.spinner("Uploading dataframe..."):
            #     name_save = f"RESULTS/benchmark/flex2023/{key}_df.csv"
            #     utils.upload_df_to_s3(df, name_save)

            # with st.spinner("Uploading edf..."):
            #     name_save = f"DATASET/flex2023/F{subject}/{pathfile.split('/')[-1]}"
            #     utils.upload_file_to_s3(pathfile, name_save)
            # st.success(f"Done upload: {name_save}")


        ## Visualization
        with tab2:
            st.markdown('''
                :blue[Check ERDS visualization. The correct response can be seen as follows:]\n
                + :blue[Left / Right --> observe strong ERD (decrease power) in alpha_C4 / alpha_C3]\n
                ''')
            
            with st.spinner("Plotting curve..."): 
                for task in ["hand", "foot"]:
                    with st.expander(f":blue[{task}]", expanded=True):
                        get_tfr(
                            tmin=TFR_TMIN, 
                            tmax=TFR_TMAX, 
                            baseline=TFR_BASELINE, 
                            task=task
                        )
                        path = plot_curve(task=task)
                        st.image(Image.open(path))

                    # with st.spinner("Upload image..."):
                    #     name_save = f"RESULTS/benchmark/flex2023/{key}_curve_{task}.png"
                    #     utils.upload_file_to_s3(path, name_save)









