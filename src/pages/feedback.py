import streamlit as st
import streamlit.components.v1 as components
import mne
from PIL import Image
import os
from src.plots import get_tfr, plot_curve, plot_heatmap
from src.ml import run_ml_feedback, plot_ml_feedback
import src.utils as utils
from src.config import *


################ MAIN #################
def write():
    """Used to write the page in the app.py file"""

    st.header("MI Feedback (for protocol 4c only)")

    ## Uploader
    with st.expander(label=':blue[MI Experiment Runs]', expanded=True):
        path_upload_processed = st.file_uploader(
            "Upload (multiple) EEG files [.edf]", 
            type=["edf"],
            accept_multiple_files=True)

    ## Check upload success
    if len(path_upload_processed) > 0:
        
        ## [Level-1] Tab_files
        list_tab_files = st.tabs([value.name for value \
                                    in path_upload_processed])
        for i, tab_file in enumerate(list_tab_files):
            with tab_file:
                
                ## data
                uploaded = path_upload_processed[i]
                pathfile = utils.save_uploaded_file(uploaded)

                fn = uploaded.name
                run = fn[fn.find("run") : fn.find("run")+4] 
                tmp = [int(i[1:]) for i in fn.split("_") \
                        if "F" in i and len(i)<5] # F13->[int(13)]
                subject = tmp[0]

                st.session_state.current_subject = subject
                st.session_state.current_run = run
                key = f"{subject}_{run}"
                st.session_state.path_edf[key] = pathfile # 13-run1

                ## [Level-2] Tab_ML & ERDS
                col1, col2 = st.columns(2)

                # ML benchmark
                with col1:
                    st.markdown('''
                        :blue[Check AUC-ROC/Accuracy of (LH-RH)/(LF-RF)/(4class) model]\n
                        :blue[Apply 3 channels C3-Cz-C4. Using CSP(8-13Hz).]
                        ''')

                    with st.spinner("Running classifier..."):
                        df = run_ml_feedback()
                        img = plot_ml_feedback(df)
                        st.image(img)

                    # ## Upload data 
                    # with st.spinner("Uploading dataframe..."):
                    #     name_save = f"RESULTS/benchmark/flex2023/{key}_df.csv"
                    #     utils.upload_df_to_s3(df, name_save)

                    # with st.spinner("Uploading edf..."):
                    #     name_save = f"DATASET/flex2023/F{subject}/{pathfile.split('/')[-1]}"
                    #     utils.upload_file_to_s3(pathfile, name_save)
                    # st.success(f"Done upload: {name_save}")


                ## Visualization
                with col2:
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



    





