import sys
import os
import time
import warnings
import shutil
warnings.filterwarnings("ignore")
import dill
from itertools import permutations, combinations, product

import scipy as sp
from scipy import random
import quantities as pq
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

import neo
from neo import NixIO
import elephant as ele

from tqdm import tqdm

from plotters import *
from functions import *
from config import get_config

tp.banner("benchmarking SeqPeelSort", 78)
tp.banner("author: Georg Raiser - grg2rsr@gmail.com", 78)

# ██ ███    ██ ██
# ██ ████   ██ ██
# ██ ██ ██  ██ ██
# ██ ██  ██ ██ ██
# ██ ██   ████ ██

# parameters for sim
n = 22
fixed_rate = 100 * pq.Hz
t_stop_sim = 1 * pq.s
tm_percentiles = sp.linspace(0, 100, n)
tm_threshs = sp.linspace(0, 1, n)

fixed_percentile = 75
fixed_threshold = 0.5

store_sorted = False

# ██████  ███████  █████  ██████
# ██   ██ ██      ██   ██ ██   ██
# ██████  █████   ███████ ██   ██
# ██   ██ ██      ██   ██ ██   ██
# ██   ██ ███████ ██   ██ ██████

# get config
config_path = os.path.abspath(sys.argv[1])

Config = get_config(config_path)
print_msg('config file read from ' + config_path)
unit_names = Config['general']['units']
exp_name = Config['general']['experiment_name']

# handling paths and creating output directory
os.chdir(os.path.dirname(config_path))
os.makedirs(os.path.join(exp_name+'_results', 'benchmark'), exist_ok=True)
sim_data_path = os.path.join(exp_name+'_results', 'benchmark', 'sim_data_params.nix')
shutil.copyfile(config_path, os.path.join(exp_name+'_results', 'benchmark', 'config.ini'))

# read in and simulate Templates
Templates_path = os.path.join(exp_name+'_results', 'templates.dill')
if not os.path.exists(Templates_path):
    print_msg('no templates found. Run SeqPeelSort first.')
    sys.exit()

with open(Templates_path, 'rb') as fH:
    Templates = dill.load(fH)
print_msg('Templates read from ' + Templates_path)

Templates_sim = {}
for unit in unit_names:
    Templates_sim[unit] = simulate_Templates(Templates[unit])[0]


# ███████ ██ ███    ███
# ██      ██ ████  ████
# ███████ ██ ██ ████ ██
#      ██ ██ ██  ██  ██
# ███████ ██ ██      ██

tm_percentile_combos = list(product(tm_percentiles, repeat=len(unit_names)))
tm_thresh_combos = list(product(tm_threshs, repeat=len(unit_names)))

Rates = []
for i in range(len(tm_percentile_combos)):
    Rates.append(dict(zip(unit_names, [fixed_rate]*2)))

print_msg("generating simulated dataset with fixed rate")
Blk = simulate_dataset(Templates_sim, Rates, Config, sim_dur=1*pq.s, save=sim_data_path)


# ███████  ██████  ██████  ████████               ████████ ██   ██ ██████  ███████ ███████ ██   ██
# ██      ██    ██ ██   ██    ██                     ██    ██   ██ ██   ██ ██      ██      ██   ██
# ███████ ██    ██ ██████     ██        █████        ██    ███████ ██████  █████   ███████ ███████
#      ██ ██    ██ ██   ██    ██                     ██    ██   ██ ██   ██ ██           ██ ██   ██
# ███████  ██████  ██   ██    ██                     ██    ██   ██ ██   ██ ███████ ███████ ██   ██

#### TM - copied from SeqPeelSort.py ####
def tm_run(AnalogSignal, templates_sim, config):
    AnalogSignal = copy.deepcopy(AnalogSignal)
    Scores_TM = template_match(AnalogSignal, templates_sim)
    SpikeTrain_TM, Score_TM = spike_detect_on_TM(
        Scores_TM, config['wsize'], percentile=config['tm_percentile'], thresh=config['tm_thresh'])
    V_peeled, V_recons = peel(AnalogSignal, SpikeTrain_TM, Scores_TM, templates_sim)

    return V_peeled, V_recons, SpikeTrain_TM, Score_TM


for j, seg in enumerate(tqdm(Blk.segments, desc='template matching segment')):

    for i, unit in enumerate(Config['general']['units']):
        config = Config[unit]
        # modified
        config['tm_percentile'] = fixed_percentile
        config['tm_thresh'] = tm_thresh_combos[j][i]

        # defining basis of peel
        if i == 0:
            Asig, = select_by_dict(seg.analogsignals, kind='original')
        else:
            previous_unit = Config['general']['units'][i-1]
            Asig, = select_by_dict(seg.analogsignals, kind='V_peeled after ' + previous_unit)

        # peeling tm runs
        V_peeled, V_recons, SpikeTrain_TM, Score_TM = tm_run(Asig, Templates_sim[unit], config)

        # annotate
        V_peeled.annotate(kind='V_peeled after ' + unit)
        V_recons.annotate(kind='V_recons after ' + unit)
        SpikeTrain_TM.annotate(kind='TM', unit=unit)
        Score_TM.annotate(kind='TM_Score', unit=unit)

        # and add
        seg.analogsignals.append(V_peeled)
        seg.analogsignals.append(V_recons)
        seg.analogsignals.append(Score_TM)
        seg.spiketrains.append(SpikeTrain_TM)
#### copy end ####


# store the sorted block?
if store_sorted:
    from neo import NixIO
    outpath = os.path.join(exp_name+'_results', 'benchmark', 'sim_data_thresholds.nix')
    with NixIO(filename=outpath) as Writer:
        print_msg("writing block containing the sorted result to " + outpath)
        Writer.write_block(Blk)
        print_msg("...done")


#  ██████  ██    ██  █████  ███    ██ ████████ ██ ███████ ██  ██████  █████  ████████ ██  ██████  ███    ██
# ██    ██ ██    ██ ██   ██ ████   ██    ██    ██ ██      ██ ██      ██   ██    ██    ██ ██    ██ ████   ██
# ██    ██ ██    ██ ███████ ██ ██  ██    ██    ██ █████   ██ ██      ███████    ██    ██ ██    ██ ██ ██  ██
# ██ ▄▄ ██ ██    ██ ██   ██ ██  ██ ██    ██    ██ ██      ██ ██      ██   ██    ██    ██ ██    ██ ██  ██ ██
#  ██████   ██████  ██   ██ ██   ████    ██    ██ ██      ██  ██████ ██   ██    ██    ██  ██████  ██   ████
#     ▀▀
# %%
Results = pd.DataFrame(columns=['unit', 'miss', 'err', 'method'] +
                       [unit_name + '_tm_thresh' for unit_name in unit_names])
for i, seg in enumerate(tqdm(Blk.segments, desc='error quantification')):
    for unit in unit_names:
        st_true, = select_by_dict(seg.spiketrains, unit=unit, kind='truth')
        st_pred_tm, = select_by_dict(seg.spiketrains, unit=unit, kind='TM')

        miss_tm, err_tm = quantify_error_rates(st_true, st_pred_tm, ttol=0.5*pq.ms)
        data_tm = [unit, miss_tm, err_tm, 'tm'] + list(tm_thresh_combos[i])
        Results = Results.append(pd.Series(data_tm, index=Results.columns), ignore_index=True)

# store the quant result
outpath = os.path.join(exp_name+'_results', 'benchmark', 'tm_thresh_sweep_results.csv')
Results.to_csv(outpath)

# ██████  ██       ██████  ████████ ████████ ██ ███    ██  ██████
# ██   ██ ██      ██    ██    ██       ██    ██ ████   ██ ██
# ██████  ██      ██    ██    ██       ██    ██ ██ ██  ██ ██   ███
# ██      ██      ██    ██    ██       ██    ██ ██  ██ ██ ██    ██
# ██      ███████  ██████     ██       ██    ██ ██   ████  ██████

# %%
# left here for future debug reasons
# Results = pd.read_csv(os.path.join(exp_name+'_results','benchmark','tm_thresh_sweep_results.csv'))

unit_combos = list(combinations(unit_names, 2))

for unit_combo in unit_combos:
    for unit in unit_combo:
        fig, axes = plt.subplots(figsize=[7.5, 3], ncols=2)

        thresh_labels = [name+'_tm_thresh' for name in unit_combo]

        for i, err_param in enumerate(['miss', 'err']):
            res = Results.groupby(('unit', 'method')).get_group((unit, 'tm'))[
                [err_param, thresh_labels[0], thresh_labels[1]]].copy()
            for j in thresh_labels:
                res[j] = [sp.around(r, 2) for r in res[j]]

            res_piv = res.pivot(thresh_labels[0], thresh_labels[1], err_param)

            sns.heatmap(res_piv[::-1], ax=axes[i], cmap='plasma', vmin=0, vmax=1, cbar=False)
            axes[i].set_aspect('equal')
            axes[i].set_title(err_param)

        fig.suptitle(unit)
        cbar = fig.colorbar(axes[0].get_children()[0], ax=axes[1])
        cbar.set_label("fraction")
        fig.tight_layout()
        # save figure
        unit_str = '_'.join(list(unit_combo)+[unit])
        outpath = os.path.join(exp_name+'_results', 'benchmark', 'thresh_sweep_' +
                               unit_str+'.'+Config['general']['fig_format'])
        fig.savefig(outpath)
        plt.close(fig)

# %%

# ███████  ██████  ██████  ████████               ██████  ███████ ██████   ██████
# ██      ██    ██ ██   ██    ██                  ██   ██ ██      ██   ██ ██
# ███████ ██    ██ ██████     ██        █████     ██████  █████   ██████  ██
#      ██ ██    ██ ██   ██    ██                  ██      ██      ██   ██ ██
# ███████  ██████  ██   ██    ██                  ██      ███████ ██   ██  ██████

print_msg("reading data from " + sim_data_path)
with NixIO(filename=sim_data_path) as Reader:
    Blk = Reader.read_block()
    print_msg("... done")

for j, seg in enumerate(tqdm(Blk.segments, desc='template matching segment')):

    for i, unit in enumerate(Config['general']['units']):
        config = Config[unit]
        config['tm_thresh'] = fixed_threshold
        config['tm_percentile'] = tm_percentile_combos[j][i]  # mod

        # defining basis of peel
        if i == 0:
            Asig, = select_by_dict(seg.analogsignals, kind='original')
        else:
            previous_unit = Config['general']['units'][i-1]
            Asig, = select_by_dict(seg.analogsignals, kind='V_peeled after ' + previous_unit)

        # peeling tm runs
        V_peeled, V_recons, SpikeTrain_TM, Score_TM = tm_run(Asig, Templates_sim[unit], config)

        # annotate
        V_peeled.annotate(kind='V_peeled after ' + unit)
        V_recons.annotate(kind='V_recons after ' + unit)
        SpikeTrain_TM.annotate(kind='TM', unit=unit)
        Score_TM.annotate(kind='TM_Score', unit=unit)

        # and add
        seg.analogsignals.append(V_peeled)
        seg.analogsignals.append(V_recons)
        seg.analogsignals.append(Score_TM)
        seg.spiketrains.append(SpikeTrain_TM)

# store the sorted block?
if store_sorted:
    from neo import NixIO
    outpath = os.path.join(exp_name+'_results', 'benchmark', 'sim_data_percentiles.nix')
    with NixIO(filename=outpath) as Writer:
        Writer.write_block(Blk)
        print_msg("output written to "+outpath)


#  ██████  ██    ██  █████  ███    ██ ████████ ██ ███████ ██  ██████  █████  ████████ ██  ██████  ███    ██
# ██    ██ ██    ██ ██   ██ ████   ██    ██    ██ ██      ██ ██      ██   ██    ██    ██ ██    ██ ████   ██
# ██    ██ ██    ██ ███████ ██ ██  ██    ██    ██ █████   ██ ██      ███████    ██    ██ ██    ██ ██ ██  ██
# ██ ▄▄ ██ ██    ██ ██   ██ ██  ██ ██    ██    ██ ██      ██ ██      ██   ██    ██    ██ ██    ██ ██  ██ ██
#  ██████   ██████  ██   ██ ██   ████    ██    ██ ██      ██  ██████ ██   ██    ██    ██  ██████  ██   ████
#     ▀▀
# %%
Results = pd.DataFrame(columns=['unit', 'miss', 'err', 'method'] +
                       [unit_name + '_tm_percentile' for unit_name in unit_names])
for i, seg in enumerate(tqdm(Blk.segments, desc='error quantification')):
    for unit in unit_names:
        st_true, = select_by_dict(seg.spiketrains, unit=unit, kind='truth')
        st_pred_tm, = select_by_dict(seg.spiketrains, unit=unit, kind='TM')

        miss_tm, err_tm = quantify_error_rates(st_true, st_pred_tm, ttol=0.5*pq.ms)
        data_tm = [unit, miss_tm, err_tm, 'tm'] + list(tm_percentile_combos[i])
        Results = Results.append(pd.Series(data_tm, index=Results.columns), ignore_index=True)

# store the quant result
outpath = os.path.join(exp_name+'_results', 'benchmark', 'tm_percentile_sweep_results.csv')
Results.to_csv(outpath)


# ██████  ██       ██████  ████████ ████████ ██ ███    ██  ██████
# ██   ██ ██      ██    ██    ██       ██    ██ ████   ██ ██
# ██████  ██      ██    ██    ██       ██    ██ ██ ██  ██ ██   ███
# ██      ██      ██    ██    ██       ██    ██ ██  ██ ██ ██    ██
# ██      ███████  ██████     ██       ██    ██ ██   ████  ██████

# left for debug reasons
# Results = pd.read_csv(os.path.join(exp_name+'_results','benchmark','tm_percentile_sweep_results.csv'))

# %%
unit_combos = list(combinations(unit_names, 2))

for unit_combo in unit_combos:
    for unit in unit_combo:
        fig, axes = plt.subplots(figsize=[7.5, 3], ncols=2)

        percentile_labels = [name+'_tm_percentile' for name in unit_combo]

        for i, err_param in enumerate(['miss', 'err']):
            res = Results.groupby(('unit', 'method')).get_group((unit, 'tm'))[
                [err_param, percentile_labels[0], percentile_labels[1]]].copy()

            for j in percentile_labels:
                res[j] = [sp.around(r, 2) for r in res[j]]

            res_piv = res.pivot(percentile_labels[0], percentile_labels[1], err_param)

            sns.heatmap(res_piv[::-1], ax=axes[i], cmap='plasma', vmin=0, vmax=1, cbar=False)
            axes[i].set_aspect('equal')
            axes[i].set_title(err_param)

        fig.suptitle(unit)
        cbar = fig.colorbar(axes[0].get_children()[0], ax=axes[1])
        cbar.set_label("fraction")
        fig.tight_layout()
        # save figure
        unit_str = '_'.join(list(unit_combo)+[unit])
        outpath = os.path.join(exp_name+'_results', 'benchmark', 'percentile_sweep_' +
                               unit_str+'.'+Config['general']['fig_format'])
        fig.savefig(outpath)
        plt.close(fig)
