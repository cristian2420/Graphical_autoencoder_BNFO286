import pandas as pd
import numpy as np
import os
import pickle as pkl
import math
from itertools import combinations
import networkx as nx


def generate_combined_chromatogram(dict_list, prominence_vals, min_peak_area=100, max_peak_area = 1e6, max_peaks_desired=2,
                                   max_peaks_allowed=4,max_scale=10, min_scale=1, max_abs_skew=20, max_violations_allowed=3,
                                   min_absolute_intensity=100):
    '''
    INPUT: A list of dictionaries for different prominence values and their respective prominence cutoffs in a list.
    
    The function will compare protein chromatograms across prominence_vals and keep the chromatogram that has the least number of violations.

    OUTPUT: A dictionary that combines chromatograms obtained from the peaks list after processing using different prominence values.
    '''
    
    for d in dict_list:
        all_proteins = []
        all_proteins+=list(d.keys())

    all_proteins = list(set(all_proteins))

    combined_chromatogram = {}
    
    for p in all_proteins:
        best_violations = float("inf")
        best_prominence = float("inf")
        
        peaks_prominence_pairs = []
        
        for i in range(len(dict_list)):
            if p in dict_list[i].keys():
                peaks_prominence_pairs.append((dict_list[i][p], i))

        protein_chances = len(peaks_prominence_pairs)

        for (a,b) in peaks_prominence_pairs:
            current_violations = count_violations(a, min_peak_area, max_peak_area, max_peaks_desired,max_peaks_allowed, max_scale, min_scale, max_abs_skew, min_absolute_intensity)
            if current_violations > max_violations_allowed:
                protein_chances-=1
                continue
            if current_violations < best_violations:
                best_violations = current_violations
                best_prominence_index = i
            if current_violations == best_violations and best_prominence_index > b: #We want the highest prominence possible to avoid overfitting
                best_violations = current_violations
                best_prominence_index = i

        if protein_chances != 0:
            combined_chromatogram[p] = dict_list[i][p]
            

    return combined_chromatogram

def count_violations(protein_dict, min_peak_area, max_peak_area, max_peaks_desired, max_peaks_allowed, max_scale, min_scale, max_abs_skew, min_absolute_intensity):

    areas_violations = 0
    peak_n = 0
    peak_n_violations = 0
    scale_violations = 0
    skew_violations = 0

    max_absolute_intensity_observed = 0 
    
    for peak in protein_dict.keys():
        if protein_dict[peak]['signal_maximum'] > max_absolute_intensity_observed:
            max_absolute_intensity_observed = protein_dict[peak]['signal_maximum']
            
        peak_n += 1
    
        if protein_dict[peak]['area'] < min_peak_area or protein_dict[peak]['area'] > max_peak_area:
            areas_violations += 1
    
        if protein_dict[peak]['scale'] < min_scale or protein_dict[peak]['scale'] > min_scale:
            scale_violations += 1
    
        if abs(protein_dict[peak]['skew']) > max_abs_skew:
            skew_violations += 1

    #These coming violations are non-negotiable. If they are violated, the function returns inf for number of violations
    if max_absolute_intensity_observed < min_absolute_intensity or peak_n > max_peaks_allowed:
        return float("inf")

    if peak_n > 4: #Add the penalization for each extra peak above the desired
        peak_n_violations=peak_n-max_peaks_desired

    return(areas_violations + peak_n_violations + scale_violations + skew_violations)

def generate_overlap_df(combined_chromatogram, n_fractions, base_delta=1,
                         reference_scale=10, abs_skew_limit=2, skew_importance=1, return_base=False,
                         no_deltas=False, first_fraction=1):

    overlaps = np.zeros((len(combined_chromatogram), n_fractions))
    current_row = 0
    key_order = []
    for idx, peaks in combined_chromatogram.items():
        key_order.append(idx)
        for p in peaks:
            positive_delta, negative_delta = calculate_deltas(combined_chromatogram[idx][p]["scale"], combined_chromatogram[idx][p]["skew"], base_delta, reference_scale, abs_skew_limit, skew_importance, return_base, no_deltas)
            centered_fraction = int(combined_chromatogram[idx][p]["retention_time"])
            left_limit, right_limit = max(centered_fraction-negative_delta, 1), min(centered_fraction+positive_delta, n_fractions)

            if positive_delta == 0 and negative_delta == 0:
                overlaps[current_row, centered_fraction] = 1
            else:
                overlaps[current_row,left_limit-1:right_limit] = 1
        current_row+=1
              
    return(pd.DataFrame(overlaps[:, first_fraction-1:], index=key_order))

def calculate_deltas(scale, skew, base_delta=1, reference_scale=10, abs_skew_limit=2, skew_importance=1, return_base=False, no_deltas=False):

    if no_deltas:
        return 0,0
        
    positive_delta = None
    negative_delta = None

    if return_base:
        return base_delta, base_delta
        
    if skew <= 0:
        if skew < -abs_skew_limit:
            negative_delta = 0
            positive_delta = base_delta
        else:
            negative_delta = int(round(base_delta + (scale/reference_scale)*base_delta+skew_importance*abs(skew)))
            positive_delta = base_delta
    if skew >= 0:
        if skew > abs_skew_limit:
            positive_delta = 0
            negative_delta = base_delta
        else:
            positive_delta = int(round(base_delta + (scale/reference_scale)*base_delta+skew_importance*abs(skew)))
            negative_delta = base_delta

    return positive_delta, negative_delta

def construct_network(overlap_df):
    '''
    OUTPUT: adjacency list describing the network
    '''
    all_edges = set()
    network = []
    indices = {}
    
    for column in overlap_df.columns:
        indices[column] = overlap_df.index[overlap_df[column] == 1].tolist()

    for column, idx_list in indices.items():
        column_edges = list(combinations(idx_list, 2))
        for a,b in column_edges:
            if a+b not in all_edges and b+a not in all_edges and a != b: #Excluding existing nodes and self-nodes
                network.append((a,b))
                all_edges.add(a+b)
                all_edges.add(b+a)

    return list(network), indices

def consensus_network(networks_list):

    consensus_network = []
    network_sets = [set(n) for n in networks_list[1:]]

    for a,b in networks_list[0]:
        for network in network_sets:
            if (a,b) in network or (b,a) in network:
                continue
            else:
                break
            consensus_network.append((a,b))

    return consensus_network

def write_adj_list(network, filename):
    '''
    INPUT: A list of tuples indicating edges
    '''

    with open(filename+".txt", "w") as f:
        for a,b in network:
            f.write(f"{a} {b}\n")
        f.close()

if __name__ == "__main__":
    
    with open("rep1_peaks.pkl", "rb") as f:
        rep1_peaks = pkl.load(f)
    with open("rep1_peaks_2.pkl", "rb") as f:
        rep1_peaks_2 = pkl.load(f)
    with open("rep1_peaks_3.pkl", "rb") as f:
        rep1_peaks_3 = pkl.load(f)

    with open("rep2_peaks.pkl", "rb") as f:
        rep2_peaks = pkl.load(f)
    with open("rep2_peaks_2.pkl", "rb") as f:
        rep2_peaks_2 = pkl.load(f)
    with open("rep2_peaks_3.pkl", "rb") as f:
        rep2_peaks_3 = pkl.load(f)

    combined_chromatogram1 = generate_combined_chromatogram([rep1_peaks, rep1_peaks_2, rep1_peaks_3], [0.10, 0.20, 0.35], min_scale=1,
                                                       min_absolute_intensity=300, max_peak_area = 100000,
                                                       max_peaks_allowed=3)
    
    combined_chromatogram2 = generate_combined_chromatogram([rep2_peaks, rep2_peaks_2, rep2_peaks_3], [0.10, 0.20, 0.35], min_scale=1,
                                                       min_absolute_intensity=300, max_peak_area = 100000,
                                                       max_peaks_allowed=3)
    
    overlap_df1 = generate_overlap_df(combined_chromatogram1,72,no_deltas=True, first_fraction=6)
    overlap_df2 = generate_overlap_df(combined_chromatogram2,72,no_deltas=True, first_fraction=6)

    network1, indices1 = construct_network(overlap_df1)
    network2, indices2 = construct_network(overlap_df2)

    final_network = consensus_network([network1, network2])

    write_adj_list(final_network, "consensus_network")
