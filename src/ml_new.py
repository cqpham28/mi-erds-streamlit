"""
python -m benchmark.8class.test
"""
import streamlit as st
import pandas as pd
import numpy as np
# dataset
from src.ml_pipeline import Pipeline_ML
from src.flex2023 import Flex2023_moabb_st
from src.formulate import Formulate
#plot
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image


SEEDS = 42
FS = 128
LIST_FILTER_BANK = [[i, i+4] for i in np.arange(4,34,2)]
CONFIG_DATA = {
    "3chan_8-14Hz": {
        "channels": ["C3", "Cz", "C4"],
        "bandpass": [[8,14]],
    },
    "allchan_8-30Hz": {
        "channels": None,
        "bandpass": [[8,30]],
    },
    "allchan_filterbank": {
        "channels": None,
        "bandpass": LIST_FILTER_BANK,
    }
}
LIST_ML_METHOD = ["CSP+LDA", "Cov+Tangent+LDA", "Cov+MDM", "FBCSP+LDA"]
T_REST = [3.0, 5.0] # t_rest in the final 2s of 5s-task



    
#==============================#
@st.cache_data
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

    for cf_key,cf_value  in CONFIG_DATA.items():
        for method in LIST_ML_METHOD:
            if ("filterbank" in cf_key and "FBCSP" not in method) \
                or ("filterbank" not in cf_key and "FBCSP" in method):
                continue
              
            dataset = Flex2023_moabb_st(protocol, session, run)
            bandpass = cf_value["bandpass"]
            channels = cf_value["channels"]
            f = Formulate(dataset, FS, subject,
                bandpass=bandpass, 
                channels=channels, 
                t_rest=(-4,-2),
                t_mi=(0, 2),
                run_to_split=None,
            )
            x, y = f.form(model_name)


            p = Pipeline_ML(method=method)
            _df = p.run(x, y, augment=False)
            r = _df['score'].mean()

            df.loc[j, "model_name"] = model_name
            df.loc[j, "subject"] = subject
            df.loc[j, "run"] = run
            df.loc[j, "config"] = str(cf_key)
            df.loc[j, "x.shape"] = str(x.shape)
            # df.loc[j, "t_rest"] = str(T_REST)
            df.loc[j, "method"] = method
            df.loc[j, "score"] = r
            j += 1

    return df



def plot_ml(df):
	""" visualization """

	g = sns.catplot(
		data=df,
		x="method",
		y="score",
		hue="config",
		kind="bar",
		palette="viridis", 
		height=3.5, aspect=3,
	)
	# iterate through axes
	for ax in g.axes.ravel():
		# add annotations
		for c in ax.containers:
			labels = [f"{v.get_height():.2f}" for v in c]
			ax.bar_label(c, labels=labels, label_type='edge')
		ax.margins(y=0.2)

	plt.ylim((0,1))
	# plt.suptitle(f"MI_2class (3chan, t=0-3s)", fontweight='bold', x=0.9)
	# plt.show()

	# plt.tight_layout()
	plt.savefig(f'refs/bar.png')
	img = Image.open(f'refs/bar.png')

	return img