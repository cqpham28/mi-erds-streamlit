"""
"""
import os
import numpy as np
import streamlit as st
from src.ml_8c import rest_ml, mi_ml
import streamlit as st



# @st.cache
def get_edf_s3(path_file:str, path_save:str):
    """get path of downloaded edf from aws s3"""
    if os.path.isfile(path_save):
        st.success(f"Detect file: {path_save}")
    else:
        st.session_state.aws["bucket"] \
            .download_file(path_file, path_save)
        st.success(f"S3 Downloaded: {path_save}")
    



def write():
    """Used to write the page in the app.py file"""

    st.header("Analysis")

    ## Choose edf file
    col1,col2,col3,col4 = st.columns([1,1,1,9])
    
    with col1:
        list_protocols = ["8c", "8c*", "4c"]
        prot = st.selectbox("Protocol", list_protocols)
    with col2:
        list_subjects = [i for i,v in st.session_state.all_files.items() \
                            if len(v)>0 and f"_{prot}_" in v[0]]
        subject = st.selectbox("Subject", list_subjects)
    with col3:
        list_sessions = ["ss1", "ss2", "ss3", "ss4"]
        session = st.selectbox("Session", list_sessions)
    with col4:
        # flatten lsit of list
        list_files = [f for f in st.session_state.all_files[subject] \
                      if "md" not in f and f.endswith(".edf") \
                        and session in f]
        try:
            fn = st.selectbox("Choose file", list_files)
            run = fn[fn.find("run") : fn.find("run")+4]
        except:
            raise FileNotFoundError("Could not find the edf files")

    ##
    button_analyze = st.button(":red[Analyze]")
    if button_analyze:

        # setup file path (protocol = prot_session)
        protocol = f"{prot}_{session}"
        key = f"F{subject}_{protocol}_{run}"

        # get data from S3
        path_save = f"refs/{key}.edf"
        get_edf_s3(path_file=path_edf, 
                   path_save=path_save)
        st.session_state.path_edf[key] = path_save

        with st.expander(f":blue[MODEL REST]", expanded=True):
            # run benchmark ML 
            df = rest_ml(protocol=protocol, 
                        subject=subject, 
                        run=run)
            st.write(df)


        with st.expander(f":blue[MODEL MI-4class]", expanded=True):
            # run benchmark ML 
            df = mi_ml(protocol=protocol, 
                        subject=subject, 
                        run=run)
            st.write(df)