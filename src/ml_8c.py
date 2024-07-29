"""
python -m benchmark.8class.test
"""

# dataset
from src.ml_pipeline import Pipeline_ML
import pandas as pd
from sklearn.metrics import classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import os
from src.flex2023 import Flex2023_moabb_st
from src.formulate import Formulate


SEEDS = 42
FS = 128
CONFIG_DATA = {
    "3chan_8-14Hz": {
        "channels": ["C3", "Cz", "C4"],
        "bandpass": [[8,14]],
    },
    "allchan_8-30Hz": {
        "channels": None,
        "bandpass": [[8,30]],
    }
}
LIST_ML_METHOD = ["CSP+LDA", "Cov+Tangent+LDA", "Cov+MDM"]
T_REST = [3.0, 5.0] # t_rest in the final 2s of 5s-task


#==============================#
def get_data(**kwargs):
    """for single file (per subject per run)"""

    dataset = Flex2023_moabb_st(protocol=kwargs["protocol"], 
                             runs=kwargs["run"])
    f = Formulate(dataset, FS, kwargs["subject"], 
                bandpass=kwargs["bandpass"],
                channels=kwargs["channels"],
                t_rest=(-4,-2),
                t_mi=(0, 2),
                run_to_split=None,
    )
    return f

        

#==============================#
def rest_ml(protocol:str="8c", subject:int=40, run:int=1):
    """
    Run model binary REST using ML feature.
    """
    df = pd.DataFrame()
    j = 0

    for _,cf_d  in CONFIG_DATA.items():
        f = get_data(protocol=protocol,
                     subject=subject, 
                     run=run,
                    bandpass=cf_d["bandpass"], 
                    channels=cf_d["channels"] 
                        )
        x, y = f.form_8c_rest(T_REST) 
        
        for method in LIST_ML_METHOD:
            p = Pipeline_ML(method=method)
            _df = p.run(x, y, augment=False)
            r = _df['score'].mean()

            df.loc[j, "model_name"] = "8c_rest"
            df.loc[j, "subject"] = subject
            df.loc[j, "run"] = run
            df.loc[j, "channels"] = str(cf_d["channels"])
            df.loc[j, "bandpass"] = str(cf_d["bandpass"])
            df.loc[j, "x.shape"] = str(x.shape)
            df.loc[j, "t_rest"] = str(T_REST)
            df.loc[j, "method"] = method
            df.loc[j, "score"] = r
            j += 1

    return df


#==============================#
def mi_ml(protocol:str="8c", subject:int=40, run:int=1):
    """
    Run model binary REST using ML feature.
    """
    df = pd.DataFrame()
    j = 0
    list_model_name = ["8c_hand", "8c_mi"]
    for _,cf_d  in CONFIG_DATA.items():
        f = get_data(protocol=protocol,
                     subject=subject, 
                     run=run,
                    bandpass=cf_d["bandpass"], 
                    channels=cf_d["channels"] 
                    )
        for model_name in list_model_name:
            if model_name == "8c_hand": 
                x, y = f.form_8c_hand()
            elif model_name == "8c_mi": 
                x, y = f.form_8c_mi()
        
            for method in LIST_ML_METHOD:
                p = Pipeline_ML(method=method)
                _df = p.run(x, y, augment=False)
                r = _df['score'].mean()

                df.loc[j, "model_name"] = model_name
                df.loc[j, "subject"] = subject
                df.loc[j, "run"] = run
                df.loc[j, "channels"] = str(cf_d["channels"])
                df.loc[j, "bandpass"] = str(cf_d["bandpass"])
                df.loc[j, "x.shape"] = str(x.shape)
                df.loc[j, "method"] = method
                df.loc[j, "score"] = r
                j += 1

    return df
