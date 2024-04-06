"""

"""
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from moabb.paradigms import MotorImagery

import mne
from mne.io import concatenate_raws
from mne.stats import permutation_cluster_1samp_test as pcluster_test
from mne.time_frequency import tfr_multitaper
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import seaborn as sns

from src.data import Flex2023_moabb_st




# @st.cache_data
def get_tfr(
    tmin = 1, 
    tmax = 9,
    baseline = (1,2),
    task = "hand",
    ):
    """
    Args:
        dataset: MNE's dataset of FLEX2023.
        event_ids: dict(left_hand=2, right_hand=1,)
    Return:
        MNE Epochs, MNE tfr
    """
    ## ADAPT STREAMLIT
    subject = st.session_state.current_subject
    run = st.session_state.current_run
    dataset = Flex2023_moabb_st()
    dataset.subject_list = [subject]
    dataset.runs = run

    ## CONFIG
    fmin, fmax = 0, 63.999
    freqs = np.arange(2, 36)  # frequency for tfr
    event_ids = st.session_state.data_run[task]["event_ids"]

    print("t crop epochs: {}s-{}s with baseline {}".format(
        tmin, tmax, baseline))

    ## get moabb paradigm
    paradigm = MotorImagery(
            events = list(event_ids.keys()), 
            n_classes = len(event_ids.keys()),
            fmin = fmin, 
            fmax = fmax, 
            ) 

    ## get RAW from moabb to compatible with mne
    X, labels, meta = paradigm.get_data(
        dataset=dataset, 
        subjects=[subject], 
        return_raws=True
        )
    raw = concatenate_raws([f for f in X])
    raw = raw.resample(sfreq=128)
    print(raw)
    print(raw.info)

    events, _ = mne.events_from_annotations(raw, event_id=event_ids)
    # print(events)

    # epochs always within (events-tmin, events+tmax)
    epochs = mne.Epochs(
        raw,
        events,
        event_ids,
        tmin-0.5,
        tmax+0.5,
        picks=("C3", "Cz", "C4"),
        baseline=None,
        preload=True,
    )


    # time-frequency representation
    tfr = tfr_multitaper(
        epochs,
        freqs=freqs,
        n_cycles=freqs,
        use_fft=True,
        return_itc=False,
        average=False,
        decim=2,
    )
    tfr.crop(tmin, tmax).apply_baseline(baseline, mode="percent")

    st.session_state.data_run[task][run] = (epochs, tfr)
    return





# @st.cache_data
def plot_heatmap(task="hand"):
    """
    Plot ERDS map (Clemen Bruner) 
    """
	## ADAPT STREAMLIT
    event_ids = st.session_state.data_run[task]["event_ids"]
    run = st.session_state.current_run
    (epochs, tfr) = st.session_state.data_run[task][run]

    vmin, vmax = -1, 1.5  # set min and max ERDS values in plot
    cnorm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)  # min, center & max ERDS
    kwargs = dict(
        n_permutations=100, 
        step_down_p=0.05, 
        seed=1, 
        buffer_size=None, 
        out_type="mask"
    )

    fig, axes = plt.subplots(
        2, 4, figsize=(17, 10), 
        gridspec_kw={"width_ratios": [10,10,10,1]}
    )
    for i, event in enumerate(event_ids):
        # select desired epochs for visualization
        tfr_ev = tfr[event]
        for ch, ax in enumerate(axes[i, :-1]):  # for each channel
            # positive clusters
            _, c1, p1, _ = pcluster_test(
                tfr_ev.data[:, ch], tail=1, **kwargs)
            # negative clusters
            _, c2, p2, _ = pcluster_test(
                tfr_ev.data[:, ch], tail=-1, **kwargs)

            # note that we keep clusters with p <= 0.05 from the combined clusters
            # of two independent tests; in this example, we do not correct for
            # these two comparisons
            c = np.stack(c1 + c2, axis=2)  # combined clusters
            p = np.concatenate((p1, p2))  # combined p-values
            mask = c[..., p <= 0.05].any(axis=-1)

            # plot TFR (ERDS map with masking)
            tfr_ev.average().plot(
                [ch],
                cmap="RdBu",
                cnorm=cnorm,
                axes=ax,
                colorbar=False,
                show=False,
                mask=mask,
                mask_style="mask",
            )
            ax.set_title(f"[{event}] {epochs.ch_names[ch]}", 
                fontsize=13, fontweight='bold')
            ax.axvline(2, linewidth=1.5, color="black", linestyle=":")  
            ax.axvline(4, linewidth=1.5, color="black", linestyle=":")  
            ax.axvline(8, linewidth=1.5, color="black", linestyle=":") 
            if ch != 0:
                ax.set_ylabel("")
                ax.set_yticklabels("")
        fig.colorbar(axes[i, 0].images[-1], cax=axes[i, -1]).ax.set_yscale("linear")
    
    # plt.suptitle(st.session_state.current_file)
    plt.tight_layout()
    plt.savefig(f'refs/heatmap_{task}.png')
    img = Image.open(f'refs/heatmap_{task}.png')

    return img


# @st.cache_data
def plot_curve(task="hand"):
    """
    Plot ERD/ERDS curve 
    """

	## ADAPT STREAMLIT
    run = st.session_state.current_run
    (epochs, tfr) = st.session_state.data_run[task][run]


    channels = ("C3", "Cz", "C4")
    df = tfr.to_data_frame(
        time_format=None, long_format=True)
  
    # Map to frequency bands:
    freq_bounds = {"_": 0, "delta": 3, "theta": 7, "alpha": 13, 
        "beta": 35, "gamma": 140}

    df["band"] = pd.cut(
        df["freq"], list(freq_bounds.values()), 
        labels=list(freq_bounds)[1:]
    )

    # Filter to retain only relevant frequency bands:
    # freq_bands_of_interest = ["alpha", "beta"]
    freq_bands_of_interest = ["alpha"]

    df = df[df.band.isin(freq_bands_of_interest)]
    df["band"] = df["band"].cat.remove_unused_categories()

    # Order channels for plotting:
    df["channel"] = df["channel"].cat.reorder_categories(
        channels, ordered=True)
    print(df)


    with sns.color_palette("Set1"):
        g = sns.FacetGrid(df, row="band", col="channel", 
            margin_titles=True, height=5)
        g.map(sns.lineplot, "time", "value", "condition", n_boot=10)
        g.add_legend(ncol=2, loc="lower center", fontsize=15)
        
        axline_kw = dict(color="black", linestyle="dashed", linewidth=1.5)
        g.map(plt.axhline, y=0, **axline_kw)
        g.map(plt.axvline, x=2, **axline_kw)
        g.map(plt.axvline, x=4, **axline_kw)
        g.map(plt.axvline, x=8, **axline_kw)

        g.set(ylim=(None, 1.5))
        g.set_axis_labels("times", "ERDS")
        g.set_titles(col_template="{col_name}", row_template="{row_name}")
        g.fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.08)

    plt.tight_layout()
    plt.savefig(f'refs/curve_{task}.png')
    img_c = Image.open(f'refs/curve_{task}.png')
    plt.close()

    return img_c


