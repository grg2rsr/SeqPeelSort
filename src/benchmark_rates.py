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

if os.name == 'posix':
    tp.banner("benchmarking SeqPeelSort", 78)
    tp.banner("author: Georg Raiser - grg2rsr@gmail.com", 78)
else:
    print("benchmarking SeqPeelSort")
    print("author: Georg Raiser - grg2rsr@gmail.com")


# ██ ███    ██ ██
# ██ ████   ██ ██
# ██ ██ ██  ██ ██
# ██ ██  ██ ██ ██
# ██ ██   ████ ██

# parameters for sim
t_stop_sim = 3  * pq.s
rate_start = 10 * pq.Hz
rate_stop = 300 * pq.Hz
n_rates = 21

store_sorted = False

rates = sp.linspace(rate_start.magnitude, rate_stop.magnitude, n_rates) * rate_start.units


# ██████  ███████  █████  ██████
# ██   ██ ██      ██   ██ ██   ██
# ██████  █████   ███████ ██   ██
# ██   ██ ██      ██   ██ ██   ██
# ██   ██ ███████ ██   ██ ██████

# get config
config_path = os.path.abspath(sys.argv[1])
# config_path = '../examples/example_config.ini'

Config = get_config(config_path)
print_msg('config file read from ' + config_path)
unit_names = Config['general']['units']
exp_name = Config['general']['experiment_name']

# handling paths and creating output directory
os.chdir(os.path.dirname(config_path))
os.makedirs(os.path.join(exp_name+'_results', 'benchmark'), exist_ok=True)
sim_data_path = os.path.join(exp_name+'_results', 'benchmark', 'sim_data_rates.nix')
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

rate_combos = list(product(rates, repeat=len(unit_names)))

Rates = []
for i in range(len(rate_combos)):
    Rates.append(dict(zip(unit_names, rate_combos[i])))

print_msg("generating simulated dataset with variable rates")
Blk = simulate_dataset(Templates_sim, Rates, Config, sim_dur=t_stop_sim, save=sim_data_path)


# ███████  ██████  ██████  ████████
# ██      ██    ██ ██   ██    ██
# ███████ ██    ██ ██████     ██
#      ██ ██    ██ ██   ██    ██
# ███████  ██████  ██   ██    ██

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

# thresholding
for seg in tqdm(Blk.segments, desc='thresholding'):
    for i, unit in enumerate(Config['general']['units']):
        config = Config[unit]
        St_pred = spike_detect(seg.analogsignals[0], config['bounds'])
        St_pred.annotate(unit=unit, kind='thresholded')
        seg.spiketrains.append(St_pred)

# store the sorted block?
if store_sorted:
    from neo import NixIO
    outpath = os.path.join(exp_name+'_results', 'benchmark', 'sim_data_rates.nix')
    with NixIO(filename=outpath) as Writer:
        Writer.write_block(Blk)
        print_msg("output written to "+outpath)


#  ██████  ██    ██  █████  ███    ██ ████████ ██ ███████ ██  ██████  █████  ████████ ██  ██████  ███    ██
# ██    ██ ██    ██ ██   ██ ████   ██    ██    ██ ██      ██ ██      ██   ██    ██    ██ ██    ██ ████   ██
# ██    ██ ██    ██ ███████ ██ ██  ██    ██    ██ █████   ██ ██      ███████    ██    ██ ██    ██ ██ ██  ██
# ██ ▄▄ ██ ██    ██ ██   ██ ██  ██ ██    ██    ██ ██      ██ ██      ██   ██    ██    ██ ██    ██ ██  ██ ██
#  ██████   ██████  ██   ██ ██   ████    ██    ██ ██      ██  ██████ ██   ██    ██    ██  ██████  ██   ████
#     ▀▀

Results = pd.DataFrame(columns=['unit', 'miss', 'err', 'method'] +
                       [unit_name + '_rate' for unit_name in unit_names])
for i, seg in enumerate(tqdm(Blk.segments, desc='error quantification')):
    for unit in unit_names:
        st_true, = select_by_dict(seg.spiketrains, unit=unit, kind='truth')
        st_pred_tm, = select_by_dict(seg.spiketrains, unit=unit, kind='TM')
        st_pred_th, = select_by_dict(seg.spiketrains, unit=unit, kind='thresholded')

        miss_tm, err_tm = quantify_error_rates(st_true, st_pred_tm, ttol=0.5*pq.ms)
        miss_th, err_th = quantify_error_rates(st_true, st_pred_th, ttol=0.5*pq.ms)

        data_tm = [unit, miss_tm, err_tm, 'tm'] + [r.magnitude for r in rate_combos[i]]
        data_th = [unit, miss_th, err_th, 'th'] + [r.magnitude for r in rate_combos[i]]

        Results = Results.append(pd.Series(data_tm, index=Results.columns), ignore_index=True)
        Results = Results.append(pd.Series(data_th, index=Results.columns), ignore_index=True)

# store the quant result
outpath = os.path.join(exp_name+'_results', 'benchmark', 'rates_sweep_result.csv')
Results.to_csv(outpath)


# ██████  ██       ██████  ████████ ████████ ██ ███    ██  ██████
# ██   ██ ██      ██    ██    ██       ██    ██ ████   ██ ██
# ██████  ██      ██    ██    ██       ██    ██ ██ ██  ██ ██   ███
# ██      ██      ██    ██    ██       ██    ██ ██  ██ ██ ██    ██
# ██      ███████  ██████     ██       ██    ██ ██   ████  ██████

Results = pd.read_csv(os.path.join(exp_name+'_results', 'benchmark', 'rates_sweep_result.csv'))

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

# %% randomly inspect single sorting results
rand_inds = random.randint(len(Blk.segments), size=5)
for i, seg in enumerate([Blk.segments[ind] for ind in rand_inds]):
    outpath = os.path.join(exp_name+'_results', 'benchmark', 'segment_'+str(rand_inds[i]))
    figures = plot_TM_result(seg, Config, zoom=0.25*pq.s, save=outpath)

    for j, unit in enumerate(unit_names):
        fig = figures[unit]
        for k, unit_ in enumerate(unit_names):
            st, = select_by_dict(seg.spiketrains, unit=unit_, kind='truth')
            for ax in fig.axes:
                plot_SpikeTrain(st, ax=ax, lw=3, color=colors[k], zorder=-10, alpha=0.2)

# %%
unit_combos = list(combinations(unit_names, 2))

for unit_combo in unit_combos:
    for unit in unit_combo:
        fig, axes = plt.subplots(figsize=[7.5, 3], ncols=2)

        rate_labels = [name+'_rate' for name in unit_combo]

        for i, err_param in enumerate(['miss', 'err']):
            res = Results.groupby(('unit', 'method')).get_group((unit, 'tm'))[
                [err_param, rate_labels[0], rate_labels[1]]].copy()
            for label in rate_labels:
                res[label] = [sp.float32(r) for r in res[label]]

            res_piv = res.pivot(rate_labels[0], rate_labels[1], err_param)
            res_piv.columns = sp.around(res_piv.columns, 2)
            res_piv.index = sp.around(res_piv.index, 2)

            sns.heatmap(res_piv[::-1], ax=axes[i], cmap='plasma', vmin=0, vmax=1, cbar=False)
            axes[i].set_aspect('equal')
            axes[i].set_title(err_param)

        fig.suptitle(unit)
        cbar = fig.colorbar(axes[0].get_children()[0], ax=axes[1])
        cbar.set_label("fraction")
        fig.tight_layout()
        # save figure
        unit_str = '_'.join(list(unit_combo)+[unit])
        outpath = os.path.join(exp_name+'_results', 'benchmark', 'rate_sweep_absolute_' +
                               unit_str+'.'+Config['general']['fig_format'])
        fig.savefig(outpath)
        plt.close(fig)

# %% difference between thresholded and tm
for unit_combo in unit_combos:
    for unit in unit_combo:
        fig, axes = plt.subplots(figsize=[7.5, 3], ncols=2)

        rate_labels = [name+'_rate' for name in unit_combo]

        for i, err_param in enumerate(['miss', 'err']):
            res_tm = Results.groupby(('unit', 'method')).get_group((unit, 'tm'))[
                [err_param, rate_labels[0], rate_labels[1]]].copy()
            res_th = Results.groupby(('unit', 'method')).get_group((unit, 'th'))[
                [err_param, rate_labels[0], rate_labels[1]]].copy()

            for label in rate_labels:
                res_tm[label] = [sp.float32(r) for r in res_tm[label]]
                res_th[label] = [sp.float32(r) for r in res_th[label]]

            res_piv_tm = res_tm.pivot(rate_labels[0], rate_labels[1], err_param)
            res_piv_tm.columns = sp.around(res_piv_tm.columns, 2)
            res_piv_tm.index = sp.around(res_piv_tm.index, 2)

            res_piv_thresh = res_th.pivot(rate_labels[0], rate_labels[1], err_param)
            res_piv_thresh.columns = sp.around(res_piv_thresh.columns, 2)
            res_piv_thresh.index = sp.around(res_piv_thresh.index, 2)

            res_diff = res_piv_tm - res_piv_thresh

            sns.heatmap(res_diff[::-1], ax=axes[i], cmap='PiYG_r', vmin=-0.25, vmax=0.25, cbar=False)
            axes[i].set_aspect('equal')
            axes[i].set_title(err_param)

        fig.suptitle(unit)
        cbar = fig.colorbar(axes[0].get_children()[0], ax=axes[1])
        cbar.set_label('TM - thresholded')

        fig.tight_layout()
        # save figure
        unit_str = '_'.join(list(unit_combo)+[unit])
        outpath = os.path.join(exp_name+'_results', 'benchmark', 'rate_sweep_comparison_' +
                               unit_str+'.'+Config['general']['fig_format'])
        fig.savefig(outpath)
        plt.close(fig)
