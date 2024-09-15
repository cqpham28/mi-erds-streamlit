import os
import pandas as pd
import streamlit as st
import src.utils as utils


ROOT = "DATASET/FLEX"


def init_key():
    """Add template _<subject_id>_<protocol>_<session>_<run_id>"""

    if "template_key" not in st.session_state.keys():
        list_subjects = [f"F{i}" for i in range(1,100)]
        list_protocols = ["4c", "8c", "8c*"]
        list_sessions = [f"ss{i}" for i in range(1,10)]
        list_runs = [f"run{i}" for i in range(1,10)]
        keys = []
        for sb in list_subjects:
            for proc in list_protocols:
                for ss in list_sessions:
                    for run in list_runs:
                        keys.append(f"{sb}_{proc}_{ss}_{run}")
        st.session_state["template_key"] = keys

def init_files():
    """load path of all available files"""

    if "all_files" not in st.session_state:
        st.session_state.all_files = {i:[] for i in range(1,100)}

        list_files = [i for i in utils.fetch_path(ROOT) \
                        if i.endswith(".edf")]
        
        for f in list_files:
            sb = f[f.find("_F")+1:].split("_")[0] # F37
            subject = int(sb[1:])
            st.session_state.all_files[subject].append(f)


def write():
    """ config """

    st.subheader("BCI")
    st.sidebar.info("This is the platform for internal usage.")

    # INIT
    utils.init_s3()
    st.success(f"Bucket Initiated")
    st.write(st.session_state.aws)
    ## Init
    init_files()
    init_key()
    
    col1, col2 = st.columns([1,2])
    
    ## Upload file option
    with col1:
        st.markdown(
            """
            :red[Upload file edf/json files onto AWS S3] 
            """
        )
        path_upload = st.file_uploader(
            "Uploader", 
            type=["edf", "json", "csv", "txt"],
            accept_multiple_files=True
        )

        if len(path_upload) > 0:
            for i,v in enumerate(path_upload):
                # get name
                uploaded = path_upload[i]
                filename = uploaded.name

                # Check whether key is correct
                key = filename[filename.find("_F")+1 : filename.find("run")+4]
                if key in st.session_state["template_key"]:

                    with st.spinner(":blue[UPLOAD TO S3...]"):
                        # save temporary local file
                        utils.save_uploaded_file(uploaded)

                        # upload to s3
                        name_source = f"refs/{filename}"
                        subject = key.split("_")[0]
                        name_save = os.path.join(ROOT, subject, filename)
                        utils.upload_file_to_s3(name_source, name_save)

                        # remove local file
                        # os.remove(name_source)
                    # log
                    st.success(f"Uploaded: {name_save}")
                
                else:
                    st.warning("[WARNING] The file should have the following: \
                                :blue[<subID>_<protocol>_<session>_<runID>] \
                                .Example: ..._F37_8c_ss1_run1_... \
                    ")
                    continue

                
    # Check Files
    with col2:
        with st.expander(":red[Check Bucket files]", expanded=False):
            df = []
            for k in st.session_state.all_files.keys():
                tmp = st.session_state.all_files[k]
                list_shorten = [fn[fn.find("_F")+1:] for fn in tmp \
                                if fn.endswith(".edf")]
                _df = pd.DataFrame.from_dict(list_shorten)
                df.append(_df)
            df = pd.concat(df, ignore_index=True)
            st.dataframe(df, width=1000) 
            # for k,v in st.session_state.all_files.items():
            #     if len(v)>0:
            #         st.write(k, v)

