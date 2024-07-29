

## FLEX
LIST_SUBJECTS =  list(range(1, 100))
EEG_CH_NAMES = [
    'Cz', 'Fz', 'Fp1', 'F7', 'F3', 
    'FC1', 'C3', 'FC5', 'FT9', 'T7', 
    'CP5', 'CP1', 'P3', 'P7', 'PO9', 
    'O1', 'Pz', 'Oz', 'O2', 'PO10', 
    'P8', 'P4', 'CP2', 'CP6', 'T8', 
    'FT10', 'FC6', 'C4', 'FC2', 'F4', 
    'F8', 'Fp2'
]
SAMPLING_RATE = 128
FILTER_NOTCH = 50.0 # Hz for notch filtering
FILTER_LOWCUT = 1.0 # Hz for lowpass


## ML
LIST_FILTER_BANKS = ([8, 12], [12, 16], [16, 20], [20, 24], [24, 28], [28, 32])
SEEDS = 42
KFOLD = 5

## PLOTS
TFR_TMIN = 1
TFR_TMAX = 9
TFR_BASELINE = (1,2)

## S3
BENCHMARK = "alphaCSP"
BLOB_BENCHMARK = f"RESULTS/benchmark/{BENCHMARK}/flex2023"



