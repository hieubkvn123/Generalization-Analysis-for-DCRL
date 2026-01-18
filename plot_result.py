import os
import json
import pathlib
import numpy as np
import matplotlib.pyplot as plt
from eval import RESULT_KEYS, COLOR_KEYS

# Enable LaTeX rendering
plt.rc('text', usetex=True)

# Visualization configs
LEGEND_FONTSIZE = 18
XYLABEL_FONTSIZE = 25
XYTICK_FONTSIZE = 15
fontconfig = {
    'family' : 'normal',
    'size' : XYLABEL_FONTSIZE
}

# Constants
SAVE_PATH = 'results/ablation_studies.pdf'
def load_json_to_dict(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def unique_everseen(seq, key=None):
    seen = set()
    seen_add = seen.add
    return [x for x, k in zip(seq,key) if not (k in seen or seen_add(k))]

def reorder_legend(ax=None,order=None,unique=False):
    if ax is None: ax=plt.gca()
    handles, labels = ax.get_legend_handles_labels()
    labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0])) # sort both labels and handles by labels
    if order is not None: # Sort according to a given list (not necessarily complete)
        keys=dict(zip(order,range(len(order))))
        labels, handles = zip(*sorted(zip(labels, handles), key=lambda t,keys=keys: keys.get(t[0],np.inf)))
    if unique:  labels, handles= zip(*unique_everseen(zip(labels,handles), key = labels)) # Keep only the first of each handle
    ax.legend(handles, labels, loc='upper left', fontsize=LEGEND_FONTSIZE)
    return(handles, labels)

def results_visualization_utils(ax, results, xaxis_data, xlabel, ylabel):
    # Initialize plot
    ax.tick_params(axis='both', which='major', labelsize=XYTICK_FONTSIZE)

    # Visualize
    for key, result in results.items():
        linestyle = None
        if key not in RESULT_KEYS.keys(): continue
        if not key.startswith('ours'):
            linestyle = '-.'
        ax.plot(xaxis_data, result, label=RESULT_KEYS[key], color=COLOR_KEYS[key], marker='o', linestyle=linestyle)
    ax.set_xlabel(xlabel, fontdict=fontconfig)
    ax.set_ylabel(ylabel, fontdict=fontconfig)

    # Save figure
    ax.grid()

if __name__ == '__main__':
    # Result files
    result_fname1 = r'results/ablation_study_depth.json'
    result_fname2 = r'results/ablation_study_width.json'

    # Read the result
    result1 = load_json_to_dict(result_fname1)
    result2 = load_json_to_dict(result_fname2)

    # Plot results
    fig, axes = plt.subplots(1, 2, figsize=(20, 7))
    results_visualization_utils(
        axes[0],
        result1['complexities'],
        result1['depths'],
        xlabel='Depths ($L$)',
        ylabel='Generalization bounds (log-scaled)',
    )
    reorder_legend(axes[0], list(RESULT_KEYS.values()))

    # --- #
    results_visualization_utils(
        axes[1],
        result2['complexities'],
        result2['widths'],
        xlabel='Widths ($W$ - in multiples of $32$)',
        ylabel='Generalization bounds (log-scaled)',
    )
    reorder_legend(axes[1], list(RESULT_KEYS.values()))

    # Plot
    plt.tight_layout()
    plt.savefig(SAVE_PATH, dpi=300, format='pdf')
