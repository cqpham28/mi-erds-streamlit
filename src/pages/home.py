import os
import streamlit as st
import src.utils as utils


def write():
    """ config """

    st.subheader("BCI")


    # Check aws info
    utils.init_s3()
    st.success(f"Bucket Initiated")
    st.write(st.session_state.aws)

    # Init
    root = "DATASET/FLEX"
    if "all_files" not in st.session_state:
        st.session_state.all_files = {i:[] for i in range(1,100)}

        list_files = [i for i in utils.fetch_path(root) \
                        if i.endswith(".edf")]
        
        for f in list_files:
            sb = int(f.split("/")[2][1:]) # ".._F37_.." -> 37  
            st.session_state.all_files[sb].append(f)

    # Upload file option
    path_upload = st.file_uploader(
        "Upload EEG files to AWS", 
        type=["edf", "json", "csv", "txt"],
        accept_multiple_files=True
    )
    if len(path_upload) > 0:
        for i,v in enumerate(path_upload):
            # get name
            uploaded = path_upload[i]
            fn = uploaded.name
            subject = fn.split("_")[3]
            
            # save temporary local file
            utils.save_uploaded_file(uploaded)

            # upload to s3
            name_source = f"refs/{fn}"
            name_save = os.path.join(root, subject, fn)
            utils.upload_file_to_s3(name_source, name_save)

            # log
            st.success(f"Uploaded: {name_save}")

                
    # Show files
    with st.expander("Check files"):
        for k,v in st.session_state.all_files.items():
            if len(v)>0:
                st.write(k, v)

