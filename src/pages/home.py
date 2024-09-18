import os
import pandas as pd
import streamlit as st
import src.utils as utils


ROOT = "DATASET/FLEX"
LIST_SUBJECTS = [f"F{i}" for i in range(1,100)]
LIST_PROTOCOLS = ["4c", "8c", "8c*"]
LIST_SESSIONS = [f"ss{i}" for i in range(1,10)]
LIST_RUNS = [f"run{i}" for i in range(1,10)]


def init_key():
    """Add template _<subject_id>_<protocol>_<session>_<run_id>"""

    if "template_key" not in st.session_state.keys():
        keys = []
        for sb in LIST_SUBJECTS:
            for proc in LIST_PROTOCOLS:
                for ss in LIST_SESSIONS:
                    for run in LIST_RUNS:
                        keys.append(f"{sb}_{proc}_{ss}_{run}")
        st.session_state["template_key"] = keys

def find_key(fullFileName:str) -> None:
    """find the config key from path of files"""

    fn = fullFileName.split("/")[-1]
    keys = [fn[fn.find("_F")+1 : fn.find("_run")+5], fn[: fn.find("run")+4]] # two types of keys
    tmpkey = [k for k in keys if k in st.session_state["template_key"]] # [F37_8c_ss1_run1]
    key = tmpkey[0]
    subject = int(key.split("_")[0][1:])

    return key, subject


def init_files():
    """load path of all available files"""

    if "all_files" not in st.session_state:
        st.session_state.all_files = {i:[] for i in range(1,100)}
        st.session_state.all_files_shorten = {i:[] for i in range(1,100)}

        list_files = [f for f in utils.fetch_path(ROOT) \
                        if "md" not in f and f.endswith(".edf")]
        
        for fn in list_files:
            ## find key and subject
            key, subject = find_key(fn)

            ## append
            st.session_state.all_files[subject].append(fn)
            fn_s = fn[fn.find(key) : ]
            st.session_state.all_files_shorten[subject].append(fn_s)



def write():
    """ config """

    st.subheader("BCI")
    st.sidebar.info("This is the platform for internal usage.")

    # INIT
    utils.init_s3()
    st.success("Bucket Initiated ([access link](%s))" % st.secrets.aws["S3_URL"])
    st.write(st.session_state.aws)
    ## Init
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
                fn = uploaded.name

                # Check whether key is contained in the filename
                key, subject = find_key(fn)
                if key is not None:

                    with st.spinner(":blue[UPLOAD TO S3...]"):
                        # save temporary local file
                        utils.save_uploaded_file(uploaded)

                        # upload to s3
                        name_source = f"refs/{fn}"
                        name_save = os.path.join(ROOT, f"F{subject}", fn)
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
        init_files()
        with st.expander("Check files", expanded=False):
            df = []
            for _,val in st.session_state.all_files_shorten.items():
                _df = pd.DataFrame.from_dict(val)
                df.append(_df)
            
            df = pd.concat(df)
            st.dataframe(df, width=1000)


