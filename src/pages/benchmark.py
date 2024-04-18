import streamlit as st
import pandas as pd
import boto3
import src.utils as utils
from src.ml import plot_benchmark



# @st.cache_data
def update_s3():
    """
    get paths & update dataframe by concatenating existing df
    """
    blob_benchmark = "RESULTS/benchmark/flex2023"

    df = []
    img = []
    for path in utils.fetch_path(blob_benchmark):
        if path.endswith(".csv"):
            df.append(utils.read_csv_from_s3(path))
        elif "curve" in path:
            img.append(path)

    st.session_state.s3_df = pd.concat(df)
    st.session_state.s3_img = img


# @st.cache_data
def init():
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
                
    col1, col2, col3 = st.columns(3)
    with col1:
        st.success(f"Bucket Initiated")
        st.write(st.session_state.aws)
    # with col2:
    #     with st.expander("Check bucket (edf)"):
    #         st.write([i for i in utils.fetch_path() \
    #             if i.endswith(".edf")])
    # with col3:
    #     with st.expander("Check bucket (other)"):
    #         pass
    #         # st.write([i for i in utils.fetch_path() \
    #         #     if i.endswith(".csv") \
    #         #     or i.endswith(".png")])




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
            col1, col2 = st.columns(2)
            
            ## Classification
            with col1:
                st.markdown('''
                    :blue[Check AUC-ROC/Accuracy of (LH-RH)/(LF-RF)/ model]\n
                    :blue[Apply 3 channels C3-Cz-C4. Using CSP(8-13Hz).]
                    ''')
                subject = float(list_subjects[i]) # 14.0
                img_bm = plot_benchmark(st.session_state.s3_df, 
                                        subject)
                st.image(img_bm)
            
            ## ERDS
            with col2:
                st.markdown('''
                    :blue[Check ERDS visualization. The correct response can be seen as follows:]\n
                    + :blue[Left / Right --> observe strong ERD (decrease power) in alpha_C4 / alpha_C3]\n
                    ''')
                list_runs = ["run1", "run2", "run3"]
                for j, _ in enumerate(st.tabs(list_runs)):
                    with _:
                        list_path = [p for p in st.session_state.s3_img \
                            if f"{int(subject)}_{list_runs[j]}" in p]
                        
                        for p in list_path:
                            st.image(utils.read_img_from_s3(p))

