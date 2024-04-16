import streamlit as st
import boto3
import pandas as pd
import src.utils as utils
from src.ml import plot_benchmark



def init():
    """ config """

    ##------S3--------##
    with st.spinner(f'Connect S3...'):
        # boto3
        session = boto3.Session(
            aws_access_key_id = st.secrets.aws["AWS_ACCESS_KEY_ID"], 
            aws_secret_access_key = st.secrets.aws["AWS_SECRET_ACCESS_KEY"]
            )
        if "aws" not in st.session_state:
            st.session_state.aws = {
                "s3": session.resource("s3"), 
                "bucket": session.resource("s3").Bucket(
                    st.secrets.aws["BUCKET_NAME"]), 
                "s3_client": session.client("s3")
                }
    st.success(f"Bucket Initiated")

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

def update_s3():
    """
    get paths & update dataframe by concatenating existing df
    """
    blob_benchmark = "RESULTS/benchmark/flex2023"
    # df = [utils.read_csv_from_s3(path) for path in \
    #         utils.fetch_path(blob_benchmark) if path.endswith(".csv")]
    # st.session_state.df = pd.concat(df)

    df = []
    img = []
    for path in utils.fetch_path(blob_benchmark):
        if path.endswith(".csv"):
            df.append(utils.read_csv_from_s3(path))
        elif "curve" in path:
            img.append(path)

    st.session_state.s3_df = pd.concat(df)
    st.session_state.s3_img = img



def write():
    """Used to write the page in the app.py file"""

    st.header("Benchmark")
    
    init()
    update_s3()

    ## check subjects
    list_subjects = [str(i) for i in \
        st.session_state.s3_df["subject"].unique()]
    
    ## plot
    for i, tab in enumerate(st.tabs(list_subjects)):
        with tab:
            col1, col2 = st.columns([1.7, 2])
            
            ## Classification
            with col1:
                subject = float(list_subjects[i]) # 14.0
                img_bm = plot_benchmark(st.session_state.s3_df, 
                                        subject)
                st.image(img_bm)
            
            ## ERDS
            with col2:
                list_runs = ["run1", "run2", "run3"]
                for j, _ in enumerate(st.tabs(list_runs)):
                    with _:
                        list_path = [p for p in st.session_state.s3_img \
                            if str(int(subject)) in p and list_runs[j] in p]
                        
                        for p in list_path:
                            st.image(utils.read_img_from_s3(p))

