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
def run_ml(
        protocol:str = "8c", 
        subject:int = 40, 
        session:str = "ss1", 
        run:str = "run1",
        model_name = "8c_mi",
    ):
    """
    Run ML model.
    """
    df = pd.DataFrame()
    j = 0

    for _,cf_d  in CONFIG_DATA.items():

        dataset = Flex2023_moabb_st(protocol, session, run)
        f = Formulate(dataset, FS, subject,
                bandpass=cf_d["bandpass"], 
                channels=cf_d["channels"], 
                t_rest=(-4,-2),
                t_mi=(0, 2),
                run_to_split=None,
        )
        
        x, y = f.form(model_name)
    
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
            df.loc[j, "t_rest"] = str(T_REST)
            df.loc[j, "method"] = method
            df.loc[j, "score"] = r
            j += 1

    return df
