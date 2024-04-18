import pandas as pd
import streamlit as st
from io import BytesIO, StringIO
import base64
from PIL import Image


################ FUNCTIONS #################
def fetch_path(prefix=""):
    """ 
    get listdir of a certain directory
    """
    list_path = []
    for obj in st.session_state.aws["bucket"].objects.filter(Prefix=prefix): 
        list_path.append(obj.key)
    return list_path



#------------------#
def read_csv_from_s3(path_file:str = ""):
    """
    Read dataframe csv file
    """
    obj = st.session_state.aws["s3"].Object(
        st.secrets.aws["BUCKET_NAME"], path_file)
    tmp = obj.get()['Body'].read()
    df = pd.read_csv(BytesIO(tmp), index_col=False)
    return df


#------------------#
def upload_df_to_s3(df, name_save):
    """
    upload dataframe (csv) to s3 bucket
    """

    stream = StringIO()
    df.to_csv(stream, index=False)
    stream.seek(0)

    st.session_state["aws"]["s3_client"].upload_fileobj(
        BytesIO(stream.read().encode()),
        st.secrets.aws["BUCKET_NAME"], 
        name_save
        )


#------------------#
def read_img_from_s3(path_file=""):
    """
    download and read image from s3
    """

    pathsave = "refs/temp.png"
    st.session_state.aws["bucket"].download_file(path_file, pathsave)

    img = Image.open(pathsave)
    return img



#------------------#
def upload_file_to_s3(name_source, name_save):
    """
    upload image to bucket
    """

    st.session_state["aws"]["s3_client"].upload_file(
        name_source,
        st.secrets.aws["BUCKET_NAME"], 
        name_save
        )