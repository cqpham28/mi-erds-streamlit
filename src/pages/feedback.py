import streamlit as st
import streamlit.components.v1 as components
import mne
import os
from src.plots import get_tfr, plot_curve, plot_heatmap
from src.ml import run_ml_feedback, plot_ml_feedback
import src.utils as utils


################ MAIN #################
def write():
    """Used to write the page in the app.py file"""

    st.header("MI Feedback")

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
                pathfile = save_uploaded_file(uploaded)

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
                tab_ml, tab_curve = st.tabs(["ML", "ERDS_Curve"])

                ## ML benchmark
                with tab_ml:
                    st.markdown('''
                        :blue[Check AUC-ROC/Accuracy of (LH-RH)/(LF-RF)/(4class) model]\n
                        :blue[Apply 3 channels C3-Cz-C4. Using CSP(8-13Hz).]
                        ''')
                    with st.spinner("Running classifier..."):
                        df = run_ml_feedback()
                        img = plot_ml_feedback(df)
                        st.image(img)

                    with st.spinner("Upload dataframe..."):
                        name_save = f"RESULTS/benchmark/flex2023/{key}_df.csv"
                        utils.upload_df_to_s3(df, name_save)
                    st.success(f"Upload to {name_save}")


                ## Visualization
                with tab_curve:
                    st.markdown('''
                        :blue[Check ERDS visualization. The correct response can be seen as follows:]\n
                        + :blue[Left Hand --> observe strong ERD (decrease power) in alpha_C4]\n
                        + :blue[Right Hand --> observe strong ERD (decrease power) in alpha_C3]
                        ''')
                    
                    with st.spinner("Plotting curve..."): 
                        for task in ["hand", "foot"]:
                            with st.expander(f":blue[{task}]", expanded=True):
                                get_tfr(tmin=1, tmax=9, baseline=(1,2), task=task)
                                img_c, decoded = plot_curve(task=task)
                                st.image(img_c)

                            with st.spinner("Upload image..."):
                                name_save = f"RESULTS/benchmark/flex2023/{key}_curve_{task}"
                                utils.upload_img_to_s3(decoded, name_save)
                            st.success(f"Upload to {name_save}")


    


@st.cache_data
def save_uploaded_file(uploaded_file, save_dir="refs")->None:
    """Saves uploaded file to a specified directory"""
    
    if uploaded_file is not None:
        file_name = uploaded_file.name
        file_path = os.path.join(save_dir, file_name)
        try:
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            print(f"File '{file_name}' saved successfully to {save_dir}")

        except FileNotFoundError:
            print(f"Error: Directory '{save_dir}' does not exist.")
            print("Please create the directory or choose an existing one.")
            
        except Exception as e:
            print(f"An error occurred while saving the file: {e}")
    else:
        print("Upload a file to save.")
    
    return file_path



