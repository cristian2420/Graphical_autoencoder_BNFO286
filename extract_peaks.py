import pandas as pd
import numpy as np
import os
from hplc.io import load_chromatogram
from hplc.quant import Chromatogram
from concurrent.futures import ProcessPoolExecutor, as_completed
import pickle

data = pd.read_csv("repl1_repl2_combined.tsv", sep = "\t", index_col=0)
data.fillna(0, inplace=True)
split_index = len(data.columns) // 2
rep1 = data.iloc[:, :split_index]
rep2 = data.iloc[:, split_index:] 

def row_to_dataframe(row_index, df, n_fractions):
    """
    Given a row index and a DataFrame, create a new DataFrame with columns 'time' and 'signal'.
    """
    if row_index not in df.index:
        raise ValueError("Row index out of range")
    
    signal_values = df.loc[row_index].values  # Extract the row values
    
    time_values = list(range(1, n_fractions))
    
    return pd.DataFrame({'time': time_values, 'signal': signal_values})

def process_chromatograms(df, row_index, prominence=0.35):
    chrom = Chromatogram(df)
    peaks_df = chrom.fit_peaks(correct_baseline=False, prominence=prominence)
    
    # Convert peaks_df into a nested dictionary
    peaks_dict = peaks_df.set_index('peak_id').to_dict(orient='index')
    
    return peaks_dict

def subprocess(i, df):
    """Processes a single protein."""
    row_df = row_to_dataframe(i, df) 
    try:
        peaks_dict = process_chromatograms(row_df, i, prominence)
        return i, peaks_dict
    except:
        return i, "error"
    

def process_replicate_parallel(df, num_workers=os.cpu_count()-1):
    """Parallelized processing of chromatograms."""
    peak_data = {}
    severe_errors = []

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(subprocess, i, df): i for i in df.index}
        
        for future in as_completed(futures):
            i, peaks_dict = future.result()
            if peaks_dict == "error":
                severe_errors.append(i)
            else:
                peak_data[i] = peaks_dict

    return peak_data, severe_errors

if __name__ == "__main__":
    rep1_peaks, errors1 = process_replicate_parallel(rep1)
    with open("rep1_peaks.pkl", "wb") as f:
        pickle.dump(rep1_peaks, f)
    with open("rep1_errors.txt", "w") as f:
        f.write("\n".join(map(str, errors1)))

    rep2_peaks, errors2 = process_replicate_parallel(rep2)
    with open("rep2_peaks.pkl", "wb") as f:
        pickle.dump(rep2_peaks, f)
    
    with open("rep2_errors.txt", "w") as f:
        f.write("\n".join(map(str, errors2)))
