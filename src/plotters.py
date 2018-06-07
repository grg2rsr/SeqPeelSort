import matplotlib.pyplot as plt
import scipy as sp
import seaborn as sns
from functions import *
import quantities as pq
import neo

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

default_time = pq.s
default_volt = pq.uV


# ██   ██ ███████ ██      ██████  ███████ ██████  ███████
# ██   ██ ██      ██      ██   ██ ██      ██   ██ ██
# ███████ █████   ██      ██████  █████   ██████  ███████
# ██   ██ ██      ██      ██      ██      ██   ██      ██
# ██   ██ ███████ ███████ ██      ███████ ██   ██ ███████

def zoom_obj(Obj, zoom, t_center=None):
    """
    Helper to zoom various neo.core objects in the temporal domain around
    t_center. If not specified, t_center it the middle of the time axis.

    Args:
        Obj: A neo.core.SpikeTrain or neo.core.AnalogSignal.
        zoom (quantities.Quantity): the size of the time window to zoom into
        t_center (quantities.Quantity): If not set, the center of the zoom is
            the middle of the time axis of Obj. If set, use this timepoint as
            the center instead.

    Returns:
        The zoomed neo object
    """

    Obj = copy.deepcopy(Obj)
    if type(Obj) == neo.core.spiketrain.SpikeTrain:
        if t_center is None:
            t_center = Obj.t_start + (Obj.t_stop - Obj.t_start)/2
        t_slice = (t_center - zoom/2, t_center + zoom/2)
        Obj = Obj.time_slice(*t_slice)
        Obj.t_start, Obj.t_stop = t_slice[0], t_slice[1]

    if type(Obj) == neo.core.analogsignal.AnalogSignal:
        if t_center is None:
            t_center = Obj.times[0] + (Obj.times[-1] - Obj.times[0])/2
        t_slice = (t_center - zoom/2, t_center + zoom/2)
        Obj = Obj.time_slice(*t_slice)
        Obj.t_start = t_slice[0]
    return Obj


def get_units_formatted(Quantity):
    """ formatting unit string for axes labels """
    return ''.join(['[', str(Quantity.units).split(' ')[1], ']'])


def plot_AnalogSignal(AnalogSignal, ax=None, rescale=True, **kwargs):
    """
    plots an neo.core.AnalogSignal

    Args:
        AnalogSignal (neo.core.AnalogSignal): The AnalogSignal to plot
        ax (matplotlib.axes): If set, plot into this axes.
            Otherwise, create a new one
        rescale (bool): rescale to default units
        **kwargs: to be passed to matplotlibs axes.plot()

    Returns:
        matplotlib.pyplot.axes
    """

    if ax is None:
        ax = plt.gca()

    if rescale:
        times = AnalogSignal.times.rescale(default_time).magnitude
        amps = AnalogSignal.rescale(default_volt).magnitude
    else:
        times = AnalogSignal.times.magnitude
        amps = AnalogSignal.magnitude

    ax.plot(times, amps, **kwargs)
    ax.set_xlabel('time ' + get_units_formatted(AnalogSignal.times))
    ax.set_ylabel('voltage ' + get_units_formatted(AnalogSignal))
    return ax


def plot_SpikeTrain(SpikeTrain, ax=None, **kwargs):
    """
    plots a neo.core.SpikeTrain

    Args:
        SpikeTrain (neo.core.SpikeTrain): The SpikeTrain to plot
        ax (matplotlib.axes): If set, plot into this axes.
            Otherwise, create a new one
        **kwargs: to be passed to matplotlibs axes.plot()

    Returns:
        matplotlib.pyplot.axes
    """

    if ax is None:
        ax = plt.gca()

    for t in SpikeTrain:
        ax.axvline(t.rescale(default_time).magnitude, **kwargs)

    return ax


# ███████ ██  ██████  ██    ██ ██████  ███████ ███████
# ██      ██ ██       ██    ██ ██   ██ ██      ██
# █████   ██ ██   ███ ██    ██ ██████  █████   ███████
# ██      ██ ██    ██ ██    ██ ██   ██ ██           ██
# ██      ██  ██████   ██████  ██   ██ ███████ ███████

def plot_spike_histogram(Blk, Config, save=None):
    """
    Plots a spike histogram of all spikes in the recording.

    Args:
        Blk (neo.core.Block): the Block
        Config (dict): the configuration dictionary
        save (str): save figure at this path if set

    Return:
        matplotlib.pyplot.Figure, matplotlib.pyplot.axes
    """

    # gather all spike amplitudes
    Spike_amps = []
    for seg in Blk.segments:
        SpikeTrain, = select_by_dict(seg.spiketrains, kind='all_spikes')
        spike_amps = SpikeTrain.waveforms.rescale(default_volt).magnitude.flatten()
        Spike_amps.append(spike_amps)
    Spike_amps = sp.concatenate(Spike_amps)

    fig, ax = plt.subplots()
    # spike amp hist
    ax.hist(spike_amps, bins=100, normed=True, color='black', alpha=0.8)
    # kde line
    sns.kdeplot(Spike_amps, ax=ax, color='r', kernel='gau', alpha=0.8)
    # bounds if present
    for i, unit in enumerate(Config['general']['units']):
        config = Config[unit]
        if config['bounds'] is not None:
            ax.axvspan(*config['bounds'].rescale(default_volt).magnitude,
                       color=colors[i], alpha=0.5, zorder=-1, label=unit)

    ax.legend()
    ax.set_ylabel('relative frequency')
    ax.set_xlabel('amplitude '+''.join(['[', str(SpikeTrain.waveforms.units).split(' ')[1], ']']))
    ax.set_title("Spike amplitude histogram")

    if save is not None:
        fig.savefig('.'.join([save, Config['general']['fig_format']]))
        plt.close(fig)
    return fig, ax


def plot_amp_reduction(pfit, frate_at_spikes, spike_amps, Config, unit, save=None):
    """
    Plots the spike amplitude reduction as a function of the units firing rate.

    Args:
        pfit (tuple): fit parameters
        frate_at_spikes (Quantity): the firing rate at the time point of spikes
        spike_amps (Quantity): the spike amplitudes
        Config (dict): the configuration dictionary
        unit (str): the name of the unit.

    Return:
        matplotlib.pyplot.Figure, matplotlib.pyplot.axes
    """

    # FIXME fix the awkward passing of both unit and Config. Requires a bit of
    # restructuring. Works as is now.

    fig, ax = plt.subplots()

    # the relationship
    ax.plot(frate_at_spikes, spike_amps, 'o', color='k', alpha=0.75)
    tvec_fit = sp.linspace(frate_at_spikes.min().magnitude, frate_at_spikes.max().magnitude, 100)
    ax.plot(tvec_fit, exp_decay(tvec_fit, *pfit), color='r', lw=2, alpha=0.8)

    # adding the bounds
    bounds = Config[unit]['bounds'].rescale(default_volt).magnitude
    i = Config['general']['units'].index(unit)
    ax.axhspan(bounds[0], bounds[1], color=colors[i], alpha=0.25, lw=0)

    # deco
    ax.set_title('spike amplitude decrease of unit '+unit)
    ax.set_xlabel('firing rate '+get_units_formatted(frate_at_spikes))
    ax.set_ylabel('spike amplitude '+get_units_formatted(spike_amps))

    if save is not None:
        fig.savefig('.'.join([save, Config['general']['fig_format']]))
        plt.close(fig)
    return fig, ax


def plot_spike_detect(Segment, Config, save=None, zoom=None):
    """
    plot the result of initial spike detection.

    Args:
        Segment (neo.core.Segment): the Segment to plot
        Config (dict): the configuration dictionary
        save (str): save figure at this path if set
        zoom (bool): plot zoomed version or full Segment

    Returns:
        matplotlib.pyplot.Figure, matplotlib.pyplot.axes
    """

    fig, ax = plt.subplots()

    # each spiketrain
    for i, unit in enumerate(Config['general']['units']):
        if Config[unit]['adaptive_threshold'] == True:
            St_choice = 'adaptive_thresholded'
        else:
            St_choice = 'thresholded'
        SpikeTrain, = select_by_dict(Segment.spiketrains, unit=unit, kind=St_choice)
        if zoom is not None:
            SpikeTrain = zoom_obj(SpikeTrain, zoom)
        plot_SpikeTrain(SpikeTrain, color=colors[i], ax=ax, alpha=0.85, lw=1)

    # the voltage
    Asig, = select_by_dict(Segment.analogsignals, kind='original')
    if zoom is not None:
        Asig = zoom_obj(Asig, zoom)
    plot_AnalogSignal(Asig, ax=ax, alpha=0.75, lw=0.9, color='k')

    # the bounds
    for i, unit in enumerate(Config['general']['units']):
        bounds_kws = dict(color=colors[i], alpha=0.25, lw=0)
        if Config[unit]['adaptive_threshold'] == True:
            adap_bound_lower, = select_by_dict(
                Segment.analogsignals, kind='adaptive_threshold_lower', unit=unit)
            adap_bound_upper, = select_by_dict(
                Segment.analogsignals, kind='adaptive_threshold_upper', unit=unit)

            if zoom is not None:
                adap_bound_lower = zoom_obj(adap_bound_lower, zoom)
                adap_bound_upper = zoom_obj(adap_bound_upper, zoom)

            tvec = adap_bound_lower.times.rescale(default_time).magnitude

            adap_bound_lower = adap_bound_lower.rescale(default_volt).magnitude[:, 0]
            adap_bound_upper = adap_bound_upper.rescale(default_volt).magnitude[:, 0]

            ax.fill_between(tvec, adap_bound_lower, adap_bound_upper, **bounds_kws)

        else:
            bounds = Config[unit]['bounds'].rescale(default_volt).magnitude
            if bounds[1] == sp.inf:
                bounds[1] = Segment.analogsignals[0].max() * 1.1
            ax.axhspan(bounds[0], bounds[1], **bounds_kws)

    # the MAD thresholding line
    from functions import MAD
    val = (MAD(Segment.analogsignals[0])*Config['general']
           ['mad_thresh']).rescale(default_volt).magnitude
    ax.axhline(val, linestyle=':', color='maroon', alpha=0.75, lw=1)

    if save is not None:
        fig.savefig('.'.join([save, Config['general']['fig_format']]))
        plt.close(fig)

    return fig, ax


def plot_Templates(templates, templates_sim, good_inds, Config, save=None):
    """
    plot the extracted templates with colored outliers, as well as the simulated
    templates.

    Args:
        templates (neo.core.AnalogSignal): the templates
        templates_sim (neo.core.AnalogSignal): the simulated templates
        good_inds (list): a list of boolean entries defining non-outliers
        Config (dict): the configuration dictionary
        save (str): save figure at this path if set

    Returns:
        matplotlib.pyplot.Figure, matplotlib.pyplot.axes
    """

    fig, axes = plt.subplots(ncols=2, sharex=True, sharey=True)
    plot_AnalogSignal(templates, color='k', alpha=0.2, lw=1, ax=axes[0], rescale=False)
    for i, line in enumerate(axes[0].lines):
        if not good_inds[i]:
            line.set_color('r')
    plot_AnalogSignal(templates_sim, color='k', alpha=0.2, lw=1, ax=axes[1], rescale=False)

    if save is not None:
        fig.savefig('.'.join([save, Config['general']['fig_format']]))
        plt.close(fig)
    return fig, axes


def plot_TM_result(Segment, Config, zoom=None, save=None):
    """
    plot the result of template matching.

    Args:
        Segment (neo.core.Segment): the Segment to plot
        Config (dict): the configuration dictionary
        zoom (bool): plot zoomed version or full Segment
        save (str): save figure at this path if set

    Returns:
        list: a list of matplotlib.pyplot.Figure, one for each unit.
    """

    figures = {}
    for i, unit in enumerate(Config['general']['units']):
        config = Config[unit]

        fig, axes = plt.subplots(nrows=3, sharex=True)
        ax_scores = plt.twinx(axes[2])

        # to plot
        Vraw, = select_by_dict(Segment.analogsignals, kind='original')
        Vrecons, = select_by_dict(Segment.analogsignals, kind='V_recons after ' + unit)
        Vpeeled, = select_by_dict(Segment.analogsignals, kind='V_peeled after ' + unit)
        Score_TM, = select_by_dict(Segment.analogsignals, kind='TM_Score', unit=unit)
        SpikeTrain, = select_by_dict(Segment.spiketrains, kind='TM', unit=unit)

        # top Vrecons
        if zoom is not None:
            Vraw = zoom_obj(Vraw, zoom)
            Vrecons = zoom_obj(Vrecons, zoom)
            Vpeeled = zoom_obj(Vpeeled, zoom)

        plot_AnalogSignal(Vraw, lw=1, color='k', alpha=0.5, ax=axes[0])
        plot_AnalogSignal(Vrecons, lw=1, color=colors[i], alpha=0.9, ax=axes[0])
        axes[0].set_title('V raw/reconstructed')
        axes[0].set_xlabel('')

        # middle
        plot_AnalogSignal(Vraw, lw=1, color='k', alpha=0.5, ax=axes[1])
        plot_AnalogSignal(Vpeeled, lw=1, color=colors[i], alpha=0.9, ax=axes[1])
        axes[1].set_title('V raw/peeled after this unit')
        axes[1].set_xlabel('')

        # bottom: scores
        if i == 0:
            Asig = copy.deepcopy(Vraw)
        else:
            previous_unit = Config['general']['units'][i-1]
            Asig, = select_by_dict(Segment.analogsignals, kind='V_peeled after ' + previous_unit)
            if zoom is not None:
                Asig = zoom_obj(Asig, zoom)

        if zoom is not None:
            Score_TM = zoom_obj(Score_TM, zoom)
            SpikeTrain = zoom_obj(SpikeTrain, zoom)

        plot_AnalogSignal(Asig, lw=1, color='k', alpha=0.5, ax=axes[2])
        plot_AnalogSignal(Score_TM, lw=1, color='r', alpha=0.5, ax=ax_scores, rescale=False)
        plt.axhline(config['tm_thresh'], lw=1, linestyle=':', color='maroon')
        plot_SpikeTrain(SpikeTrain, ax=axes[2], color='red')
        axes[2].set_title('V peeled before (or raw if first unit) / score')

        # deco
        ax_scores.set_ylabel('TM score')
        fig.suptitle(unit)

        if save is not None:
            outpath = save+'_'+unit+'.'+Config['general']['fig_format']
            fig.savefig(outpath)
            plt.close(fig)

        figures[unit] = fig

    return figures
