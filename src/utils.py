import pandas as pd
import streamlit as st
from io import BytesIO, StringIO
import base64
from PIL import Image


################ FUNCTIONS #################
def fetch_path(prefix):
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

    """
    obj = st.session_state.aws["s3"].Object(
        st.secrets.aws["BUCKET_NAME"], path_file)
    tmp = obj.get()['Body'].read()

    pathsave = 'refs/decoded_img.png' 
    with open(pathsave, 'wb') as f:
        decoded_image_data = base64.decodebytes(tmp)
        f.write(decoded_image_data)
    
    img = Image.open(pathsave)
    return img




#------------------#
def upload_img_to_s3(decoded, name_save):
    """
    upload image to bucket
    """

    st.session_state["aws"]["s3_client"].upload_fileobj(
        BytesIO(decoded),
        st.secrets.aws["BUCKET_NAME"], 
        name_save
        )
    