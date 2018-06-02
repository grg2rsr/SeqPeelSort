
# ██ ███    ███ ██████   ██████  ██████  ████████ ███████
# ██ ████  ████ ██   ██ ██    ██ ██   ██    ██    ██
# ██ ██ ████ ██ ██████  ██    ██ ██████     ██    ███████
# ██ ██  ██  ██ ██      ██    ██ ██   ██    ██         ██
# ██ ██      ██ ██       ██████  ██   ██    ██    ███████

# system
import sys, os, time, copy
import resource
import warnings
from tqdm import tqdm

# sci
import scipy as sp
from scipy import stats, signal, random
from scipy.optimize import curve_fit
import quantities as pq
import cv2

# ml
import sklearn
from sklearn.neighbors import LocalOutlierFactor
from sklearn.decomposition import PCA

# ephys
import neo
import elephant as ele

# print
import colorama
import tableprint as tp

warnings.filterwarnings("ignore")
t0 = time.time()


# ██   ██ ███████ ██      ██████  ███████ ██████  ███████
# ██   ██ ██      ██      ██   ██ ██      ██   ██ ██
# ███████ █████   ██      ██████  █████   ██████  ███████
# ██   ██ ██      ██      ██      ██      ██   ██      ██
# ██   ██ ███████ ███████ ██      ███████ ██   ██ ███████

def print_msg(msg,log=True):
    """
    pretty print, with elapsed time and current memory usage

    Args:
        msg (str): the string to print
    """
    # TODO check for compatibility with windows / mac systems
    mem_used = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    mem_used = sp.around(mem_used, 2)
    memstr = '('+str(mem_used) + ' GB): '
    print(colorama.Fore.CYAN + tp.humantime(time.time()-t0) + ' elapsed\t' + memstr + '\t' + colorama.Fore.GREEN + msg)
    if log:
        with open('log.log','a+') as fH:
            log_str = tp.humantime(time.time()-t0) + ' elapsed\t' + memstr + '\t' + msg + os.newline
            fH.writelines(log_str)
    pass

def select_by_dict(objs, **selection):
    """newline
    selects elements in a list of neo objects with annotations matching the
    selection dict

    Args:
        objs (list): a list of neo objects that have annotations
        selection (dict): a dict containing key-value pairs for selection

    Returns:
        list: a list containing the matching neo objects
    """
    res = []
    for obj in objs:
        if selection.items() <= obj.annotations.items():
            res.append(obj)
    return res


def get_spike_inds(SpikeTrain):
    """
    get the indices of spike times relative to an AnalogSignal with equal
    sampling rate.

    Args:
        SpikeTrain (neo.core.SpikeTrain): the SpikeTrain

    Returns:
        list: a list of indices
    """
    SpikeTrain = copy.deepcopy(SpikeTrain)
    SpikeTrain -= SpikeTrain.t_start
    SpikeTrain.t_start = 0*pq.s
    inds = sp.int32((SpikeTrain * SpikeTrain.sampling_rate).simplified)
    return inds


def MAD(AnalogSignal):
    """ median absolute deviation of an AnalogSignal """
    X = AnalogSignal.magnitude
    mad = sp.median(sp.absolute(X - sp.median(X))) * AnalogSignal.units
    return mad


def exp_decay(x, A, tau, b):
    y = A * sp.exp(-x/tau) + b
    return y


def time2ind(tvec, time):
    """
    get the index of time in time vector

    Args:
        tvec (Quantitiy): a time vector with n time points of shape (n,)
        time (Quantity): a time point

    Returns:
        int: the index
    """
    ind = sp.argmin(sp.absolute(tvec - time))
    return ind


def times2inds(tvec, ttup):
    """ index tuple, see time2ind """
    return [time2ind(tvec, t) for t in ttup]


def refractory_correct_SpikeTrain(SpikeTrain, ref_per=2*pq.ms):
    """
    remove spike duplets from a spike train that are closer together than
    ref_per, keep the earlier

    Args:
        SpikeTrain (neo.core.SpikeTrain): the SpikeTrain
        ref_per (Quantity): the refractory period - minimum time between two
            spikes

    Returns:
        neo.core.SpikeTrain: the corrected SpikeTrain
    """

    Tx,Ty = sp.meshgrid(SpikeTrain.times,SpikeTrain.times)
    T = (Tx < Ty+ref_per)
    try:
        good_inds = sp.unique(sp.argmin(T,axis=1))
        return SpikeTrain[good_inds]
    except ValueError: # if empty
        return SpikeTrain

# ████████ ██   ██ ██████  ███████ ███████ ██   ██  ██████  ██      ██████  ██ ███    ██  ██████
#    ██    ██   ██ ██   ██ ██      ██      ██   ██ ██    ██ ██      ██   ██ ██ ████   ██ ██
#    ██    ███████ ██████  █████   ███████ ███████ ██    ██ ██      ██   ██ ██ ██ ██  ██ ██   ███
#    ██    ██   ██ ██   ██ ██           ██ ██   ██ ██    ██ ██      ██   ██ ██ ██  ██ ██ ██    ██
#    ██    ██   ██ ██   ██ ███████ ███████ ██   ██  ██████  ███████ ██████  ██ ██   ████  ██████

def bounded_threshold(SpikeTrain, bounds):
    """
    removes all spike from a SpikeTrain which amplitudes do not fall within the
    specified static bounds.

    Args:
        SpikeTrain (neo.core.SpikeTrain): The SpikeTrain
        bounds (Quantity): a Quantity of shape (2,) with (lower,upper) bounds,
            unit [uV]

    Returns:
        neo.core.SpikeTrain: the resulting SpikeTrain
    """
    SpikeTrain = copy.deepcopy(SpikeTrain)
    peak_amps = SpikeTrain.waveforms.max(axis=1)

    good_inds = sp.logical_and(peak_amps > bounds[0], peak_amps < bounds[1])
    SpikeTrain = SpikeTrain[good_inds.flatten()]
    return SpikeTrain


def adaptive_threshold(SpikeTrain, adaptive_thresh_lower, adaptive_thresh_upper):
    """
    removes all spike from a SpikeTrain which amplitudes do not fall within the
    specified variable bounds.

    Args:
        SpikeTrain (neo.core.SpikeTrain): The SpikeTrain
        adaptive_thresh_lower (neo.core.AnalogSignal): the variable lower bound
        adaptive_thresh_upper (neo.core.AnalogSignal): the variable upper bound

    Returns:
        neo.core.SpikeTrain: the resulting SpikeTrain
    """
    SpikeTrain = copy.deepcopy(SpikeTrain)
    peak_amps = SpikeTrain.waveforms.max(axis=1)

    spike_inds = get_spike_inds(SpikeTrain)
    spike_amps = SpikeTrain.waveforms.max(axis=1)

    good_inds = sp.logical_and(adaptive_thresh_lower.magnitude[spike_inds] < spike_amps.magnitude,
                               adaptive_thresh_upper.magnitude[spike_inds] > spike_amps.magnitude)
    SpikeTrain = SpikeTrain[good_inds.flatten()]

    return SpikeTrain


# ███████ ██████  ██ ██   ██ ███████     ██████  ███████ ████████ ███████  ██████ ████████
# ██      ██   ██ ██ ██  ██  ██          ██   ██ ██         ██    ██      ██         ██
# ███████ ██████  ██ █████   █████       ██   ██ █████      ██    █████   ██         ██
#      ██ ██      ██ ██  ██  ██          ██   ██ ██         ██    ██      ██         ██
# ███████ ██      ██ ██   ██ ███████     ██████  ███████    ██    ███████  ██████    ██

def spike_detect(AnalogSignal, bounds, lowpass_freq=1000*pq.Hz):
    """
    detects all spikes in an AnalogSignal that fall within amplitude bounds

    Args:
        AnalogSignal (neo.core.AnalogSignal): the waveform
        bounds (Quantity): a Quantity of shape (2,) with (lower,upper) bounds,
            unit [uV]
        lowpass_freq (Quantity): cutoff frequency for a smoothing step before
            spike detection, unit [Hz]

    Returns:
        neo.core.SpikeTrain: the resulting SpikeTrain
    """

    # filter to avoid multiple peaks
    if lowpass_freq is not None:
        AnalogSignal = ele.signal_processing.butter(AnalogSignal, lowpass_freq=lowpass_freq)

    # find relative maxima / minima
    peak_inds = signal.argrelmax(AnalogSignal)[0]

    # to data structure
    peak_amps = AnalogSignal.magnitude[peak_inds, :, sp.newaxis] * AnalogSignal.units

    tvec = AnalogSignal.times
    SpikeTrain = neo.core.SpikeTrain(tvec[peak_inds], t_start=AnalogSignal.t_start,
                                     t_stop=AnalogSignal.t_stop, sampling_rate=AnalogSignal.sampling_rate, waveforms=peak_amps)

    # subset detected SpikeTrain by bounds
    SpikeTrain = bounded_threshold(SpikeTrain, bounds)

    return SpikeTrain


# ████████ ███████ ███    ███ ██████  ██       █████  ████████ ███████ ███████
#    ██    ██      ████  ████ ██   ██ ██      ██   ██    ██    ██      ██
#    ██    █████   ██ ████ ██ ██████  ██      ███████    ██    █████   ███████
#    ██    ██      ██  ██  ██ ██      ██      ██   ██    ██    ██           ██
#    ██    ███████ ██      ██ ██      ███████ ██   ██    ██    ███████ ███████

def get_templates(AnalogSignal, SpikeTrain, wsize=4*pq.ms, N=None):
    """
    Slices an AnalogSignal with the spike times of a SpikeTrain and extracts the
    waveforms in a symmetric window of specified size.

    Args:
        AnalogSignal (neo.core.AnalogSignal): the signal containing the spike
            waveforms, the recording
        SpikeTrain (neo.core.SpikeTrain): the SpikeTrain
        wsize (Quantity): temporal size of the window of the extracted waveforms
        N (int): if specified, only return N template waveforms.

    Returns:
        neo.core.AnalogSignal: the extracted template waveforms
    """
    # this function is sensitive to a rounding error bug?
    # https://github.com/NeuralEnsemble/python-neo/issues/530

    # subset spikes to only those whose window won't overlap with end of the data
    # shave off half window size on each end
    tvec = AnalogSignal.times
    valid_time = (tvec[0] + wsize/2, tvec[-1] - wsize/2)
    SpikeTrain = SpikeTrain.time_slice(*valid_time)
    SpikeTrain.t_start, SpikeTrain.t_stop = tvec[0], tvec[-1]  # to restore original time bounds

    # only get N templates
    if N is not None:
        rand_inds = sp.random.randint(SpikeTrain.shape[0], size=N)
        SpikeTrain = SpikeTrain[rand_inds]

    # store Templates AnalogSignal
    Templates = []
    for t in SpikeTrain:
        Asig = AnalogSignal.time_slice(t-wsize/2, t+wsize/2)
        Asig.t_start = -wsize/2
        Templates.append(Asig.magnitude)

    Templates = sp.concatenate(Templates, axis=1) * AnalogSignal.units
    Templates = neo.core.AnalogSignal(
        Templates, sampling_rate=AnalogSignal.sampling_rate, t_start=-wsize/2)

    return Templates


def simulate_Templates(Templates, n_sim=100, n_comp=5):
    """
    performs a PCA on the Template waveforms of specified number of components.
    Estimates the distribution along the PCs via KDA and samples a specified
    number of new values, which are transformed back to the original
    template dimensionality by an inverse PCA.

    Args:
        Templates (neo.core.AnalogSignal): the extracted real templates
        n_sim (int): number of templates to simulate
        n_comp (int): number of PCs

    Returns:
        neo.core.AnalogSignal: the resulting simulated Templates
    """

    # make an array out of the templates
    Templates_Mat = Templates.magnitude

    # make pca from templates
    pca = PCA(n_components=n_comp)
    pca.fit(Templates_Mat.T)
    pca_templates = pca.transform(Templates_Mat.T)

    # parameterize PCA result to distribution
    Shape_dist = stats.gaussian_kde(pca_templates.T)

    # generate simulated spike shape from distribution
    Templates_sim_Mat = pca.inverse_transform(Shape_dist.resample(n_sim).T).T

    Templates_sim = neo.core.AnalogSignal(
        Templates_sim_Mat, units=Templates.units, sampling_rate=Templates.sampling_rate, t_start=Templates.t_start)

    return Templates_sim, pca


def clean_Templates(Templates):
    """
    remove outlier template waveforms programatically.

    Args:
        Templates (neo.core.AnalogSignal): the extracted template waveforms

    Returns:
        (neo.core.AnalogSignal): the cleaned template waveforms
    """

    templates = Templates.magnitude
    clf = LocalOutlierFactor(n_neighbors=20)
    good_inds = clf.fit_predict(templates.T) == 1

    Templates_cleaned = neo.core.AnalogSignal(
        templates[:, good_inds], units=Templates.units, t_start=Templates.t_start, sampling_rate=Templates.sampling_rate)

    return Templates_cleaned, good_inds


#  █████  ███    ███ ██████      ██████  ███████ ██████  ██    ██  ██████ ████████ ██  ██████  ███    ██
# ██   ██ ████  ████ ██   ██     ██   ██ ██      ██   ██ ██    ██ ██         ██    ██ ██    ██ ████   ██
# ███████ ██ ████ ██ ██████      ██████  █████   ██   ██ ██    ██ ██         ██    ██ ██    ██ ██ ██  ██
# ██   ██ ██  ██  ██ ██          ██   ██ ██      ██   ██ ██    ██ ██         ██    ██ ██    ██ ██  ██ ██
# ██   ██ ██      ██ ██          ██   ██ ███████ ██████   ██████   ██████    ██    ██  ██████  ██   ████

def calc_frate_at_spikes(SpikeTrain):
    """
    estimates local firing rate at time points of spikes by convolution with a
    gaussian kernel

    Args:
        SpikeTrain (neo.core.SpikeTrain): the input SpikeTrain

    Returns:
        (Quantity): the firing rates at the time point of spikes
    """

    # estimate firing rate
    frate = ele.statistics.instantaneous_rate(
        SpikeTrain, kernel='auto', sampling_period=SpikeTrain.sampling_period)

    # get firing rate at the time points of spikes
    inds = get_spike_inds(SpikeTrain)
    frate_at_spikes = frate.magnitude[inds] * frate.units

    return frate_at_spikes


def calc_spike_amp_reduction(frate_at_spikes, spike_amps, N=1000):
    """
    estimates the reduction of spike peak amplitude as a function of the units
    firing rate by fitting the firing rate - peak amplitude relation to an
    exponetial decay function.

    Args:
        frate_at_spikes (Quantity): the firing rates at the time point of
            spikes. Output of the function calc_frate_at_spikes.
        spike_amps (Quantity): the amplitude of the spikes waveforms.
        N (int): If defined, subsample all spikes to this number for the fit.

    Return:
        (list): the parameters of best fit
    """

    params = (spike_amps.max().magnitude, frate_at_spikes.mean().magnitude, spike_amps.min().magnitude)
    bounds = ((spike_amps.min().magnitude, 0, 0),
              (spike_amps.max().magnitude, sp.inf, spike_amps.max().magnitude))

    if frate_at_spikes.shape[0] > N:
        inds = sp.random.randint(0, frate_at_spikes.shape[0], size=N)
    else:
        inds = range(frate_at_spikes.shape[0])
    try:
        pfit = curve_fit(
            exp_decay, frate_at_spikes.magnitude[inds], spike_amps.magnitude[inds], p0=params, bounds=bounds)[0]
        return pfit
    except RuntimeError:
        return None


def calc_adaptive_threshold(frate, pfit, bounds):
    """
    calculates the adaptive threshold, based on the current firing rate, the
    estimated peak reduction and the initially user set static bounds that are
    to be adaptively modified.

    Args:
        frate (neo.core.AnalogSignal): the firing rate
        pfit (list): the parameters of the fit
        bounds (Quantity): the initial static bounds that are to be modified.

    Returns:
        neo.core.AnalogSignal, neo.core.AnalogSignal: lower and upper bound,
            modified accoring to spike amp decrease
    """

    pfit_lower = (pfit[0], pfit[1], bounds[0].magnitude)
    pfit_upper = (pfit[0], pfit[1], bounds[1].magnitude)

    adap_bound_lower = exp_decay(frate.magnitude, *pfit_lower)
    adap_bound_upper = exp_decay(frate.magnitude, *pfit_upper)

    adap_bound_lower = neo.core.AnalogSignal(
        adap_bound_lower, units=bounds[0].units, t_start=frate.t_start, sampling_rate=frate.sampling_rate)
    adap_bound_upper = neo.core.AnalogSignal(
        adap_bound_upper, units=bounds[1].units, t_start=frate.t_start, sampling_rate=frate.sampling_rate)

    return adap_bound_lower, adap_bound_upper


# ████████ ███████ ███    ███ ██████  ██       █████  ████████ ███████     ███    ███  █████  ████████  ██████ ██   ██
#    ██    ██      ████  ████ ██   ██ ██      ██   ██    ██    ██          ████  ████ ██   ██    ██    ██      ██   ██
#    ██    █████   ██ ████ ██ ██████  ██      ███████    ██    █████       ██ ████ ██ ███████    ██    ██      ███████
#    ██    ██      ██  ██  ██ ██      ██      ██   ██    ██    ██          ██  ██  ██ ██   ██    ██    ██      ██   ██
#    ██    ███████ ██      ██ ██      ███████ ██   ██    ██    ███████     ██      ██ ██   ██    ██     ██████ ██   ██

def template_match(AnalogSignal, Templates_sim):
    """
    performs a template match of each simulated template waveform to the
    AnalogSignal.

    Args:
        AnalogSignal (neo.core.AnalogSignal): the AnalogSignal for the template
            match
        Templates_sim (neo.core.AnalogSignal): the simulated templates

    Returns:
        neo.core.AnalogSignal: the resulting scores of the template
            match
    """
    # TODO OPTIMIZE - multithread template match

    N = Templates_sim.shape[1]
    wsize = Templates_sim.shape[0]

    # prep
    Scores = sp.zeros((AnalogSignal.shape[0], N)).astype('float32')
    data_ = AnalogSignal.magnitude.astype('float32')
    Templates_sim = Templates_sim.magnitude.astype('float32')
    Npad = wsize-1

    # template match run
    for i in range(N):
        res = cv2.matchTemplate(data_, Templates_sim[:, i], method=1).flatten()
        Scores[:, i] = sp.pad(res, (0, Npad), mode='constant', constant_values=1)
    Scores = 1-Scores # remap from 0 to 1

    # to neo object
    Scores = neo.core.AnalogSignal(Scores, units=pq.dimensionless,
                                   t_start=AnalogSignal.times[0], sampling_rate=AnalogSignal.sampling_rate, kind='Scores')
    return Scores


def spike_detect_on_TM(Scores, wsize, percentile=90, thresh=0.5):
    """
    predict spikes based on the result of the result from the template match.

    Args:
        Scores (neo.core.AnalogSignal): the result of the template matching step
        wsize (Quantity): the size of the templates
        percentile (float): 0-100
        thresh (float): 0-1

    Returns:
        SpikeTrain (neo.core.SpikeTrain): The estimated SpikeTrain
        Score_lim (neo.core.AnalogSignal): the ...
    """

    # AnalogSignal of Score percentile
    Score_lim = neo.core.AnalogSignal(sp.percentile(
        Scores, percentile, axis=1), units=pq.dimensionless, t_start=Scores.times[0], sampling_rate=Scores.sampling_rate)

    # NOTE not necessary, but left in here in case of future adaptations. Might
    # be useful when templates are very ambiguous
    # smooth Scores_lim to avoid muliple detections
    # Score_lim = ele.signal_processing.butter(Score_lim,lowpass_freq=100*pq.Hz)

    # find peaks
    SpikeTrain = spike_detect(Score_lim, [thresh, 1.01]*pq.dimensionless, lowpass_freq=None)

    # remove doublets in case
    SpikeTrain = refractory_correct_SpikeTrain(SpikeTrain)

    # correction for template offset
    if SpikeTrain.shape[0] > 0:
        if (SpikeTrain[-1] + wsize/2 > SpikeTrain.t_stop):
            # remove last spike if won't fit in
            SpikeTrain = SpikeTrain[:-1]
        SpikeTrain += wsize/2

    return SpikeTrain, Score_lim


# ██████  ███████ ███████ ██
# ██   ██ ██      ██      ██
# ██████  █████   █████   ██
# ██      ██      ██      ██
# ██      ███████ ███████ ███████

def peel(AnalogSignal, SpikeTrain, Scores, templates_sim):
    """
    Subtracts spikes from a template based on best template match and returns
    the AnalogSignal without those spikes

    Args:
        AnalogSignal (neo.core.AnalogSignal): the recording
        SpikeTrain (neo.core.SpikeTrain): the detected spikes in the recording
        Scores (neo.core.AnalogSignal): the output of spike_detect_on_TM
        templates_sim (neo.core.AnalogSignal): the simulated templates

    Returns:
        V_remain (neo.core.AnalogSignal): the input AnalogSignal after
            subtraction
        V_recons (neo.core.AnalogSignal): the reconstructed voltage by placing
            best fit template on the time point of the spikes in SpikeTrain
    """

    # wsize fix
    wsize = templates_sim.duration.simplified
    SpikeTrain = SpikeTrain.copy() - wsize/2

    # get best matching template per spike
    inds = get_spike_inds(SpikeTrain)
    best_template_inds = sp.argmax(Scores, axis=1)[inds]

    # ini empty stuff
    V_recons = neo.core.AnalogSignal(sp.zeros(
        Scores.times.shape[0]), units=templates_sim.units, t_start=Scores.times[0], sampling_rate=Scores.sampling_rate)

    # NOTE lin fix related lines commented out. Not necessary (actually
    # decreases sorting performance) but left in in case of future
    # implementations
    # V_linfix = neo.core.AnalogSignal(sp.zeros(Scores.times.shape[0]), units=templates_sim.units, t_start=Scores.times[0],sampling_rate=Scores.sampling_rate)

    for i, t in enumerate(SpikeTrain):
        best_template = templates_sim[:, best_template_inds[i]]
        # .time_slice() method can not be used here, because it returns a view
        # wow this sucks in uglyness - the additional sampling period added is needed cause otherwise window is 1 too short ...
        try:
            inds = times2inds(V_recons.times, V_recons.time_slice(t, t+wsize).times[[0, -1]])
            inds[1] +=1
        except ValueError:
            import pdb
            pdb.set_trace()
        V_recons[slice(*inds)] += best_template
        # V_linfix[slice(*inds)] += neo.core.AnalogSignal(sp.linspace(best_template[0].magnitude,best_template[-1].magnitude,best_template.shape[0]) * best_template.units,t_start=t,sampling_period=best_template.sampling_period)

    V_remain = AnalogSignal.copy() - V_recons  # + V_linfix
    # V_remain = AnalogSignal.copy() - V_recons + V_linfix

    return V_remain, V_recons


# ██████  ███████ ███    ██  ██████ ██   ██ ███    ███  █████  ██████  ██   ██
# ██   ██ ██      ████   ██ ██      ██   ██ ████  ████ ██   ██ ██   ██ ██  ██
# ██████  █████   ██ ██  ██ ██      ███████ ██ ████ ██ ███████ ██████  █████
# ██   ██ ██      ██  ██ ██ ██      ██   ██ ██  ██  ██ ██   ██ ██   ██ ██  ██
# ██████  ███████ ██   ████  ██████ ██   ██ ██      ██ ██   ██ ██   ██ ██   ██

def generate_V_sim(Templates,rates,Config,t_stop_sim,ref_corr=False):
    """
    Args:
        Templates (dict): A dict containing a neo.core.AnalogSignal per unit
        rates (dict): vector of rates (Quantity) for each unit.
        Config (dict):
        t_stop_sim (Quantity): max time of each simulated spike train

    Returns:
        V_sim (neo.core.AnalogSignal):
        SpikeTrains_true (dict):

    Templates: dict
    rates: a dict with rates for each unit """

    unit_names = Config['general']['units']
    fs = Templates[unit_names[0]].sampling_rate

    # ini emtpy voltage with noise level inferred from templates
    # this likely slightly overestimates
    noise_sd = (Templates[unit_names[0]] - Templates[unit_names[0]].mean(axis=1)[:,sp.newaxis]).std(axis=1).std()
    V_sim = neo.core.AnalogSignal(sp.randn(int((fs*t_stop_sim).simplified.magnitude)) *  noise_sd,t_stop=t_stop_sim,sampling_rate=fs,units=Templates[unit_names[0]].units)

    # V_sim filling with spikes
    SpikeTrains_true = {}
    for i, unit in enumerate(Config['general']['units']):
        config = Config[unit]

        # generate SpikeTrain
        rate = rates[unit]
        SpikeTrain = ele.spike_train_generation.homogeneous_poisson_process(t_stop=t_stop_sim, rate=rate)
        SpikeTrain.sampling_rate = fs

        # remove unrealistic spikes
        if ref_corr:
            SpikeTrain = refractory_correct_SpikeTrain(SpikeTrain)

        # remove too too early too late spikes
        wsize = config['wsize']
        good_inds = sp.logical_and(SpikeTrain.times > wsize/2, SpikeTrain.times  < t_stop_sim-wsize/2)
        SpikeTrain = SpikeTrain[good_inds]

        # store
        SpikeTrains_true[unit] = SpikeTrain

        # fill V_sim with spikes
        SpikeTrain_ = SpikeTrain.copy() - wsize/2
        n_templates = Templates[unit].shape[1]

        tvec = V_sim.times # for speedup
        # for i,t in enumerate(tqdm(SpikeTrain_)):
        for i,t in enumerate(SpikeTrain_):
            template = Templates[unit][:,random.randint(n_templates)]
            # NOTE .time_slice() method can not be used here, because it returns a view
            inds = times2inds(tvec, V_sim.time_slice(t,t+wsize).times[[0,-1]])
            inds[1] += 1
            V_sim[slice(*inds)] += template

    return V_sim, SpikeTrains_true

def simulate_dataset(Templates, Rates, Config, sim_dur=1*pq.s, save=None):
    """
    creates a block with
    Args:
        rates: a list of dicts
    """
    # create simulated dataset
    Blk = neo.core.Block()

    # unit_names = Config['general']['units']

    for i,rates in enumerate(tqdm(Rates)):

        # simulate a recording
        V_sim, SpikeTrains_true = generate_V_sim(Templates, rates, Config, sim_dur)

        V_sim.annotate(kind='original')
        seg = neo.core.Segment()
        seg.analogsignals.append(V_sim)
        for unit, st in SpikeTrains_true.items():
            st.annotate(unit=unit,kind='truth',rate=rates[unit])
            seg.spiketrains.append(st)

        Blk.segments.append(seg)

    # write block to disk
    if save is not None:
        sim_data_path = save
        from neo import NixIO
        with NixIO(filename=sim_data_path) as Writer:
            print_msg("writing simulated data to " + sim_data_path)
            Writer.write_block(Blk)
            print_msg("... done")

    return Blk

def quantify_error_rates(st_true,st_pred,ttol=0.5*pq.ms):
    # FIXME REWRITE this docstring!!
    """ definition:
    spike is in

    true and in pred = correctly predicted = true positive
    true but not in pred = missed spike = false negative
    not in true, but in pred = erronously predicted = false positive

    true negative doesn't really make sense, hence doesn't exist

    all within ttol

    only one spike, if multiple is counted as wrong as well
    """

    # false negative counting
    miss = 0.0
    total = float(len(st_true)) # shape
    for t in st_true:
        if sp.sum(sp.absolute(st_pred.times - t) < ttol) != 1:
            miss += 1

    try:
        fmiss = miss / total # fraction of missed spikes
    except ZeroDivisionError:
        fmiss = sp.nan

    err = 0.0
    total = float(len(st_pred))
    for t in st_pred:
        if sp.sum(sp.absolute(st_true.times - t) < ttol) != 1:
            err += 1
    try:
        ferr = err / total # frac of spikes that are erronously predicted of total predicted spikes
    except ZeroDivisionError:
        ferr = sp.nan

    return fmiss, ferr
