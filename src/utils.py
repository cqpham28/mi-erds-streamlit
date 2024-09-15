import pandas as pd
import streamlit as st
from io import BytesIO, StringIO
import numpy as np
import os
from PIL import Image
import hmac
import boto3


################ FUNCTIONS #################
def fetch_path(prefix=""):
    """ 
    get listdir of a certain directory
    """
    list_path = []
    for obj in st.session_state.aws["s3"] \
                                .Bucket(st.secrets.aws["BUCKET_NAME"]) \
                                .objects.filter(Prefix=prefix): 
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


#------------------#
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





####################################
def check_password():
    """Returns `True` if the user had a correct password."""

    def login_form():
        """Form with widgets to collect user information"""
        with st.form("Credentials"):
            st.text_input("Username", key="username")
            st.text_input("Password", type="password", key="password")
            st.form_submit_button("Log in", on_click=password_entered)

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["username"] in st.secrets[
            "passwords"
        ] and hmac.compare_digest(
            st.session_state["password"],
            st.secrets.passwords[st.session_state["username"]],
        ):
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Don't store the username or password.
            del st.session_state["username"]
        else:
            st.session_state["password_correct"] = False

    # Return True if the username + password is validated.
    if st.session_state.get("password_correct", False):
        return True

    # Show inputs for username + password.
    login_form()
    if "password_correct" in st.session_state:
        st.error("😕 User not known or password incorrect")
    return False



################ AWS #################
def init_s3():
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
                # "bucket": session.resource("s3").Bucket(st.secrets.aws["BUCKET_NAME"]), 
                "s3_client": session.client("s3")
                }
            

# @st.cache
def get_edf_s3(path_file:str, path_save:str):
    """get path of downloaded edf from aws s3"""

    if os.path.isfile(path_save):
        st.success(f"Detect file: {path_save}")
    else:
        with st.spinner(":blue[DOWNLOADING FILE...]"):
            st.session_state.aws["s3"] \
                .Bucket(st.secrets.aws["BUCKET_NAME"]) \
                .download_file(path_file, path_save)
        st.success(f"S3 Downloaded: {path_save}")


        

################ SELECT FILES #################
def select_box_to_file(list_protocols=["8c", "8c*", "4c"]):
    """
    Create 4 streamlit select_box, allow user to navigate 
    through protocol->subject->session->file run

    """

    ## Choose edf file
    col1,col2,col3,col4,_ = st.columns([1,1,1,2,5])
    
    # Choose protocol
    with col1:
        protocol = st.selectbox("Protocol", list_protocols)

    # Choose available subject (from key of st.session_state.all_files)
    with col2:
        list_subjects = [i for i,v in st.session_state.all_files.items() \
                            if len(v)>0 and f"_{protocol}_" in v[0]]
        subject = st.selectbox("Subject", list_subjects)
    
    # Choose available session
    tmp_files = st.session_state.all_files[subject]
    with col3:
        list_sessions = np.unique([f[f.find("_ss")+1: f.find("_ss")+4] \
                                    for f in tmp_files])
        session = st.selectbox("Session", list_sessions)

    # Choose available file
    with col4:
        # flatten list of list
        list_files = [f for f in tmp_files \
                        if "md" not in f and f.endswith(".edf") \
                            and session in f]
        # map filename_short (F37_ss1_run1) to filename_fullpath (DATA/F37/...)
        d_fn = {
            f[f.find("_F")+1 : f.find("run")+4] : f \
                for f in list_files
        }
                
        fn = st.selectbox("Choose file", d_fn.keys())
        run = fn.split("_")[-1]
        path_file = d_fn[fn]
    
    return protocol, subject, session, run, path_file