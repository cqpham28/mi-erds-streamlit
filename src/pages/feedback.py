import streamlit as st
import streamlit.components.v1 as components
import mne
from io import BytesIO
import os

from src.plots import get_tfr, plot_curve, plot_heatmap





################ MAIN #################
def write():
    """Used to write the page in the app.py file"""

    st.header("ERDS Visualization (demo)")

    ## Uploader
    with st.expander(label=':blue[MI Experiment Runs]', expanded=True):
        path_upload_processed = st.file_uploader("Upload (multiple) EEG files [.edf]", 
                                            type=["edf"],
                                            accept_multiple_files=True)
    

    ## Check upload success
    if len(path_upload_processed) > 0:

        list_tab_files = st.tabs([value.name for value in path_upload_processed])

        for i, tab_file in enumerate(list_tab_files):
            with tab_file:
                
                ## data
                uploaded = path_upload_processed[i]
                # pathfile = save_uploaded_file(uploaded)

                fn = uploaded.name
                run_idx = int(fn[fn.find("run")+3])
                st.session_state.current_file = fn
                st.session_state.current_run = run_idx
                st.session_state.path_edf[run_idx] = fn
                # st.session_state.path_edf[run_idx] = pathfile
                
                ##
                get_tfr()
                
                ## Plot
                tab1, tab2 = st.tabs(["Curve", "Heatmap"])
                with tab1:
                    img_curve = plot_curve()
                    st.image(img_curve)
                with tab2:
                    img_heatmap = plot_heatmap()
                    st.image(img_heatmap)







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