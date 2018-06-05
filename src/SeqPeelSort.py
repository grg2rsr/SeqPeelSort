import sys
import os
import copy
import dill
import warnings

import scipy as sp
import neo
from neo import NixIO

from functions import *
from plotters import *
from config import get_config

from tqdm import tqdm

tp.banner("This is SeqPeelSort v1.0.0", 78)
tp.banner("author: Georg Raiser - grg2rsr@gmail.com", 78)


# ██ ███    ██ ██
# ██ ████   ██ ██
# ██ ██ ██  ██ ██
# ██ ██  ██ ██ ██
# ██ ██   ████ ██

# get config
config_path = os.path.abspath(sys.argv[1])

Config = get_config(config_path)
print_msg('config file read from ' + config_path)

# handling paths and creating output directory
os.chdir(os.path.dirname(config_path))
data_path = Config['general']['data_path']

os.makedirs('results', exist_ok=True)
os.makedirs(os.path.join('results', 'plots'), exist_ok=True)

# read data
with NixIO(filename=data_path) as Reader:
    Blk = Reader.read_block()

Blk.name = Config['general']['experiment_name']
print_msg('data read from ' + data_path)


# ██████  ██████  ███████ ██████  ██████   ██████   ██████ ███████ ███████ ███████
# ██   ██ ██   ██ ██      ██   ██ ██   ██ ██    ██ ██      ██      ██      ██
# ██████  ██████  █████   ██████  ██████  ██    ██ ██      █████   ███████ ███████
# ██      ██   ██ ██      ██      ██   ██ ██    ██ ██      ██           ██      ██
# ██      ██   ██ ███████ ██      ██   ██  ██████   ██████ ███████ ███████ ███████

print_msg('preprocessing data')

# always: annotate
for seg in Blk.segments:
    seg.analogsignals[0].annotate(kind='original')

# highpass filter
for seg in Blk.segments:
    seg.analogsignals[0] = ele.signal_processing.butter(
        seg.analogsignals[0], highpass_freq=Config['general']['highpass_freq'])

# invert if peaks are negative
if Config['general']['peak_mode'] == 'negative':
    for seg in Blk.segments:
        seg.analogsignals[0] *= -1


# ███████ ██████  ██ ██   ██ ███████     ██████  ███████ ████████ ███████  ██████ ████████
# ██      ██   ██ ██ ██  ██  ██          ██   ██ ██         ██    ██      ██         ██
# ███████ ██████  ██ █████   █████       ██   ██ █████      ██    █████   ██         ██
#      ██ ██      ██ ██  ██  ██          ██   ██ ██         ██    ██      ██         ██
# ███████ ██      ██ ██   ██ ███████     ██████  ███████    ██    ███████  ██████    ██

print_msg('initial spike detect')

# detecting all spikes by MAD thresholding
mad_thresh = Config['general']['mad_thresh']

for seg in Blk.segments:
    AnalogSignal, = select_by_dict(seg.analogsignals, kind='original')
    Spikes_all = spike_detect(
        AnalogSignal, [MAD(AnalogSignal)*mad_thresh, sp.inf] * AnalogSignal.units)
    Spikes_all.annotate(kind='all_spikes')
    seg.spiketrains.append(Spikes_all)

# plot
if Config['general']['fig_format'] is not None:
    outpath = 'results/plots/spike_histogram'
    plot_spike_histogram(Blk, Config, save=outpath)

# quit here if no bounds are set
if sp.any([Config[unit]['bounds'] == None for unit in Config['general']['units']]):
    print_msg("some bounds are not set. Quitting. Set all bounds in config and rerun.")
    sys.exit()

# sort into units according to thresholds
for i, unit in enumerate(Config['general']['units']):
    config = Config[unit]
    for seg in Blk.segments:
        st, = select_by_dict(seg.spiketrains, kind='all_spikes')
        Spikes = bounded_threshold(st, config['bounds'])
        Spikes.annotate(unit=unit, kind='thresholded')
        seg.spiketrains.append(Spikes)

# adaptive thresholding
for i, unit in enumerate(Config['general']['units']):
    config = Config[unit]
    if config['adaptive_threshold'] == True:
        print_msg("adaptive thresholding for unit "+unit)

        # gathering all spikes and their amplitudes
        frate_at_spikes = []
        spike_amps = []
        for i, seg in enumerate(Blk.segments):
            SpikeTrain, = select_by_dict(seg.spiketrains, unit=unit, kind='thresholded')
            frate_at_spikes.append(calc_frate_at_spikes(SpikeTrain))
            spike_amps.append(SpikeTrain.waveforms.max(axis=1))

        frate_at_spikes = sp.concatenate(frate_at_spikes, axis=0)[:, 0] * frate_at_spikes[0].units
        spike_amps = sp.concatenate(spike_amps, axis=0)[:, 0] * spike_amps[0].units

        # fitting exp decay to this
        pfit = calc_spike_amp_reduction(frate_at_spikes, spike_amps)

        # plot
        if Config['general']['fig_format'] is not None:
            outpath = 'results/plots/spike_amp_decrease_'+unit
            plot_amp_reduction(pfit, frate_at_spikes, spike_amps, Config, unit, save=outpath)

        if pfit is not None:
            for i, seg in enumerate(Blk.segments):
                # estimate firing rate
                SpikeTrain, = select_by_dict(seg.spiketrains, unit=unit, kind='thresholded')
                kernel = ele.kernels.GaussianKernel(10*pq.ms)
                frate = ele.statistics.instantaneous_rate(
                    SpikeTrain, kernel='auto', sampling_period=SpikeTrain.sampling_period)

                # calc adaptive bounds
                adap_bound_lower, adap_bound_upper = calc_adaptive_threshold(
                    frate, pfit, config['bounds'])
                adap_bound_lower.annotate(unit=unit, kind='adaptive_threshold_lower')
                adap_bound_upper.annotate(unit=unit, kind='adaptive_threshold_upper')

                # add to segment
                for adap_bound in (adap_bound_lower, adap_bound_upper):
                    seg.analogsignals.append(adap_bound)

                # calculate adaptive thresholded SpikeTrain
                SpikeTrain, = select_by_dict(seg.spiketrains, kind='all_spikes')
                SpikeTrain_adap = adaptive_threshold(SpikeTrain, adap_bound_lower, adap_bound_upper)
                SpikeTrain_adap.annotate(unit=unit, kind='adaptive_thresholded')
                seg.spiketrains.append(SpikeTrain_adap)

        else:
            # setting config to static thresholds
            print_msg("adaptive thresholding failed for unit "+unit)
            Config[unit]['adaptive_threshold'] == False

# plot
if Config['general']['fig_format'] is not None:
    for i, seg in enumerate(Blk.segments):
        outpath = 'results/plots/thresholded_'+str(i)
        plot_spike_detect(seg, Config, save=outpath, zoom=Config['general']['zoom'])


# ████████ ███████ ███    ███ ██████  ██       █████  ████████ ███████ ███████
#    ██    ██      ████  ████ ██   ██ ██      ██   ██    ██    ██      ██
#    ██    █████   ██ ████ ██ ██████  ██      ███████    ██    █████   ███████
#    ██    ██      ██  ██  ██ ██      ██      ██   ██    ██    ██           ██
#    ██    ███████ ██      ██ ██      ███████ ██   ██    ██    ███████ ███████

print_msg('getting templates ... ')

Templates = {}
Templates_sim = {}

for i, unit in enumerate(Config['general']['units']):
    config = Config[unit]
    if config['adaptive_threshold'] == True:
        St_choice = 'adaptive_thresholded'
    else:
        St_choice = 'thresholded'
    T = []
    for seg in tqdm(Blk.segments, desc='getting templates for unit '+unit):
        st, = select_by_dict(seg.spiketrains, kind=St_choice, unit=unit)
        Asig, = select_by_dict(seg.analogsignals, kind='original')
        T.append(get_templates(Asig, st, wsize=config['wsize']))

    # combine and subset
    templates = sp.concatenate([t.magnitude for t in T], axis=1) * T[0].units
    if config['n_templates'] < templates.shape[1]:
        templates = templates[:, :config['n_templates']]

    Templates[unit] = neo.core.AnalogSignal(
        templates, t_start=T[0].t_start, sampling_rate=T[0].sampling_rate)
print_msg('... done.')

# cleaning templates
print_msg('cleaning templates')

Templates_cleaned = {}
Templates_good_inds = {}
for i, unit in enumerate(Config['general']['units']):
    config = Config[unit]
    Templates_cleaned[unit], Templates_good_inds[unit] = clean_Templates(Templates[unit])

# simulating templates
print_msg('simulating templates')

for i, unit in enumerate(Config['general']['units']):
    config = Config[unit]
    Templates_sim[unit] = simulate_Templates(
        Templates_cleaned[unit], n_sim=config['n_sim'], n_comp=config['n_comp'])[0]

    # plot templates
    if Config['general']['fig_format'] is not None:
        outpath = os.path.join('results', 'plots', 'Templates_'+unit)
        plot_Templates(Templates[unit], Templates_sim[unit],
                       Templates_good_inds[unit], Config, save=outpath)

# save templates
with open(os.path.join('results', 'templates.dill'), 'wb') as fH:
    dill.dump(Templates_cleaned, fH)


# ████████ ███████ ███    ███ ██████  ██       █████  ████████ ███████     ███    ███  █████  ████████  ██████ ██   ██
#    ██    ██      ████  ████ ██   ██ ██      ██   ██    ██    ██          ████  ████ ██   ██    ██    ██      ██   ██
#    ██    █████   ██ ████ ██ ██████  ██      ███████    ██    █████       ██ ████ ██ ███████    ██    ██      ███████
#    ██    ██      ██  ██  ██ ██      ██      ██   ██    ██    ██          ██  ██  ██ ██   ██    ██    ██      ██   ██
#    ██    ███████ ██      ██ ██      ███████ ██   ██    ██    ███████     ██      ██ ██   ██    ██     ██████ ██   ██

print_msg('template matching ... ')

def tm_run(AnalogSignal, templates_sim, config):
    AnalogSignal = copy.deepcopy(AnalogSignal)
    Scores_TM = template_match(AnalogSignal, templates_sim)
    SpikeTrain_TM, Score_TM = spike_detect_on_TM(
        Scores_TM, config['wsize'], percentile=config['tm_percentile'], thresh=config['tm_thresh'])
    V_peeled, V_recons = peel(AnalogSignal, SpikeTrain_TM, Scores_TM, templates_sim)

    return V_peeled, V_recons, SpikeTrain_TM, Score_TM


for seg in tqdm(Blk.segments, desc='template matching segment'):

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

print_msg("... done.")

# plot
if Config['general']['fig_format'] is not None:
    for i, seg in enumerate(Blk.segments):
        outpath = 'results/plots/TM_result_'+str(i)
        plot_TM_result(seg, Config, zoom=None, save=outpath)
        outpath = os.path.join('results', 'plots', 'TM_result_zoomed_'+str(i))
        plot_TM_result(seg, Config, zoom=Config['general']['zoom'], save=outpath)


# ██     ██ ██████  ██ ████████ ███████      ██████  ██    ██ ████████
# ██     ██ ██   ██ ██    ██    ██          ██    ██ ██    ██    ██
# ██  █  ██ ██████  ██    ██    █████       ██    ██ ██    ██    ██
# ██ ███ ██ ██   ██ ██    ██    ██          ██    ██ ██    ██    ██
#  ███ ███  ██   ██ ██    ██    ███████      ██████   ██████     ██

# clean
Res_segs = []
for seg in Blk.segments:
    res_seg = neo.core.Segment()
    [res_seg.spiketrains.append(st) for st in select_by_dict(seg.spiketrains, kind="TM")]
    Res_segs.append(res_seg)
Blk.segments = Res_segs

for chx in Blk.channel_indexes:
    chx.analogsignals = []

output_format = Config['general']['output_format']


if output_format == 'nix':
    outpath = os.path.join('results', os.path.splitext(
        os.path.basename(data_path))[0]+"_sorted.nix")
    with NixIO(filename=outpath) as Writer:
        Writer.write_block(Blk)
        print_msg("output written to "+outpath)

if output_format == 'csv':
    import pandas as pd
    for i, seg in enumerate(Blk.segments):
        for j, st in enumerate(seg.spiketrains):
            outpath = os.path.join('results', os.path.splitext(os.path.basename(data_path))[
                                   0]+"_unit_"+st.annotations['unit']+"_trial_"+str(i)+"_sorted.csv")
            pd.Series(st.times).to_csv(path=outpath)
            print_msg("output written to "+outpath)
