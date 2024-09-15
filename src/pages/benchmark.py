import streamlit as st
import pandas as pd
import boto3
import src.utils as utils
from src.ml import plot_benchmark
from src.config import *



# @st.cache_data
def update_s3():
    """
    get paths & update dataframe by concatenating existing df
    """

    df = []
    img = []
    for path in utils.fetch_path(BLOB_BENCHMARK):
        if path.endswith(".csv"):
            df.append(utils.read_csv_from_s3(path))
        elif "curve" in path:
            img.append(path)

    st.session_state.s3_df = pd.concat(df)
    st.session_state.s3_img = img





def write():
    """Used to write the page in the app.py file"""

    st.header("Benchmark")


    # init()
    update_s3()

    ## check subjects
    list_subjects = [str(i) for i in \
        st.session_state.s3_df["subject"].unique()]
    
    ## plot
    for i, tab in enumerate(st.tabs(list_subjects)):
        with tab:
            col1, col2 = st.columns(2)
            
            # ## Classification
            # with col1:
            #     st.markdown('''
            #         :blue[Check AUC-ROC/Accuracy of (LH-RH)/(LF-RF)/ model]\n
            #         :blue[Apply 3 channels C3-Cz-C4. Using CSP(8-13Hz).]
            #         ''')
            #     subject = float(list_subjects[i]) # 14.0
            #     img_bm = plot_benchmark(st.session_state.s3_df, 
            #                             subject)
            #     st.image(img_bm)
            
            # ## ERDS
            # with col2:
            #     st.markdown('''
            #         :blue[Check ERDS visualization. The correct response can be seen as follows:]\n
            #         + :blue[Left / Right --> observe strong ERD (decrease power) in alpha_C4 / alpha_C3]\n
            #         ''')
            #     list_runs = ["run1", "run2", "run3"]
            #     for j, _ in enumerate(st.tabs(list_runs)):
            #         with _:
            #             list_path = [p for p in st.session_state.s3_img \
            #                 if f"{int(subject)}_{list_runs[j]}" in p]
                        
            #             for p in list_path:
            #                 st.image(utils.read_img_from_s3(p))

