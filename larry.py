import sys
import resampy

import numpy as np
import scipy.special as spspec
import math
import warnings

from functools import partial
from tqdm import tqdm
from time import time as give_time


def butter_lowpass_filt(N, fmax, data, sr):
    """
    Apply a lowpass Butterworth filter to the given data.

    Parameters
    ----------
    N : int
        The order of the filter.
    fmax : float
        The critical frequency of the filter in Hz.
    data : array_like
        The input signal data to be filtered.
    sr : float
        The sampling rate of the data in Hz.

    Returns
    -------
    y : ndarray
        The filtered output signal.
    """
    from scipy.signal import butter, lfilter

    b, a = butter(N=N, Wn=fmax, btype="low", fs=sr)

    return lfilter(b, a, data)


def gaussian(mu, sigma, t):
    """Compute a gaussian function with mean and std `mu` and `sigma`.

    Parameters
    ----------
    mu : float
        Center of the gaussian.
    sigma : float
        Standard deviation of the gaussian.
    t : array-like
        Time, in the same units as `mu` and `sigma`.

    Returns
    -------
    x : array-like
        Gaussian function evaluated at `t`.
    """
    x = np.exp(-(((t - mu) / (np.sqrt(2) * sigma)) ** 2))
    x = x / np.amax(x)
    return x


def phi(alpha, t):
    # center_t = t[int(len(t)/2)]
    # t2 = t-center_t # center t array around 0
    x = 0.5 * (1 + spspec.erf(alpha * t / np.sqrt(2)))
    return x


def running_mean(x, M):
    """
    Apply an M-point moving-average filter to the input array x.

    Parameters
    ----------
    x : array_like
        The input array to be filtered.
    M : int
        The width (in samples) of the moving-average window.

    Returns
    -------
    xavg : ndarray
        The filtered array.

    Notes
    -----
    This is an old function that should be re-written.
    """
    xavg = np.zeros(len(x))

    for ind in range(M, len(x) - M):
        start_ind = ind - math.floor(
            M / 2
        )  # utiliser int au lieu de math.floor devrait marcher aussi
        end_ind = start_ind + M
        xavg[ind] = np.mean(x[start_ind:end_ind])

    # What value on the left?
    for ind in range(M):
        xavg[ind] = xavg[M]

    # What value on the right?
    for ind in range(len(x) - M, len(x)):
        xavg[ind] = xavg[len(x) - M - 1]

    return xavg


def fade(inputsig, dur_fadein, dur_fadeout, fadetype, datatype):
    """
    Apply a fade in and a fade out of specific duration onto the input signal.

    Parameters
    ----------
    inputsig : array_like
        The raw input signal.
    dur_fadein : int
        The duration of the fade-in, in samples.
    dur_fadeout : int
        The duration of the fade-out, in samples.
    fadetype : {'lin', 'log'}
        The type of fade in and out. 'lin' for a linear slope between 0 and 1 (or 1 and 0),
        'log' for a logarithmic slope between 0 and 1 (or 1 and 0).
    datatype : data-type
        The type of data in the numpy array `Amplitudes` (e.g., `np.float32`), to avoid
        increasing the size of resulting sounds (default datatype for numpy is `float64`).

    Returns
    -------
    outputsig : ndarray
        The signal with applied fades.
    """
    amplitudes = np.ones(len(inputsig), dtype=datatype)

    if fadetype == "lin":
        amplitudes[0 : int(dur_fadein)] = np.linspace(0, 1, int(dur_fadein))
        amplitudes[len(inputsig) - int(dur_fadeout) : len(inputsig)] = np.linspace(
            1, 0, num=int(dur_fadeout)
        )
    elif fadetype == "log":
        amplitudes[0 : int(dur_fadein)] = np.logspace(-100, 0, num=int(dur_fadein))
        amplitudes[len(inputsig) - int(dur_fadeout) : len(inputsig)] = np.logspace(
            0, -100, num=int(dur_fadeout)
        )

    outputsig = amplitudes * inputsig

    return outputsig


def normalization(x):
    """Max-normalize the input `x`.

    Parameters
    ----------
    x : numpy.ndarray
        Input to be normalized.

    Returns
    -------
    x_norm : numpy.ndarray
        `x` normalized by its maximum absolute value.
    """
    x = x - x.mean()
    norm = np.abs(x).max()
    if norm > 0.0:
        return x / norm
    else:
        return x


def make_soundtrack(
    catalog,
    waveforms,
    num_speakers,
    movie_duration,
    real_event_dur,
    speed_factor,
    sampling_rate,
    audio_sr,
    energy_min=None,
    energy_max=None,
    column_for_scaling="energy",
    scaling_fun="lin",
    soundtrack_duration="fixed",
):
    """
    Generate a soundtrack by resampling, normalizing, and concatenating waveforms based on event catalog.

    Parameters
    ----------
    catalog : pandas.DataFrame
        The event catalog containing event metadata, including timestamps and scaling attributes.
    waveforms : dict
        Dictionary mapping event IDs to their corresponding waveform data arrays.
    num_speakers : int
        The number of speaker channels to generate.
    movie_duration : float
        The duration of the movie in seconds.
    real_event_dur : float
        The real duration of each event in seconds.
    speed_factor : float
        The factor by which to adjust the speed of the waveforms.
    sampling_rate : float
        The sampling rate of the input waveforms in Hz.
    audio_sr : float
        The desired output sampling rate in Hz.
    energy_min : float, optional
        The minimum energy value for scaling. If None, it is computed from the catalog. Default is None.
    energy_max : float, optional
        The maximum energy value for scaling. If None, it is computed from the catalog. Default is None.
    column_for_scaling : str, optional
        The column name in the catalog used for scaling. Default is "energy".
    scaling_fun : str or callable, optional
        The scaling function to use. Can be 'lin', 'log', or a custom callable. Default is 'lin'.
    soundtrack_duration : {'fixed', 'flexible'}, optional
        If 'fixed', the soundtrack duration is fixed and events that exceed the duration are excluded.
        If 'flexible', the soundtrack duration adjusts to fit all events. Default is 'fixed'.

    Returns
    -------
    tracks : ndarray
        The generated multi-speaker soundtrack, with shape (num_speakers, int(movie_duration * audio_sr)).

    Notes
    -----
    This function was designed for very specific needs of the Seismic Sound Lab.
    """

    filenotfoundlist = []

    sampling_rate_accelerated_signal_hz = sampling_rate * speed_factor
    if (
        float(int(sampling_rate_accelerated_signal_hz))
        != sampling_rate_accelerated_signal_hz
    ):
        warnings.warn(
            "Sampling rate * speed_factor cannot be unambigously"
            " interpreted as an integer. It will be rounded down to the"
            " nearest integer."
        )
    sampling_rate_accelerated_signal_hz = int(sampling_rate_accelerated_signal_hz)

    max_starttime = np.amax(catalog["timestamp"] - catalog["timestamp"][0])
    movie_starttimes = (
        (catalog["timestamp"] - catalog["timestamp"][0])
        * movie_duration
        / max_starttime
    )
    movie_starttimes = round_time(movie_starttimes, audio_sr)
    accelerated_event_dur = real_event_dur / speed_factor
    accelerated_event_dur = round_time(accelerated_event_dur, audio_sr)

    if soundtrack_duration == "fixed":
        valid_starttimes = (movie_starttimes + accelerated_event_dur) < movie_duration
        catalog = catalog[valid_starttimes]
        movie_starttimes = movie_starttimes[valid_starttimes]
    elif soundtrack_duration == "flexible":
        movie_duration = movie_starttimes.max() + accelerated_event_dur

    tracks = np.zeros((num_speakers, int(movie_duration * audio_sr)), dtype=np.float32)
    tracks_Mb = tracks.nbytes / 1024.0**2
    print(f"You are about to use {tracks_Mb:.2f}Mb")

    if energy_min is None:
        energy_min = np.min(catalog[column_for_scaling])
    if energy_max is None:
        energy_max = np.max(catalog[column_for_scaling])

    # resample
    print("Resampling event waveforms...")
    tstart = give_time()
    data_rsp = []
    valid_evids = []
    target_event_duration_samp = int(real_event_dur * sampling_rate)
    for evid in catalog.index:
        if evid in waveforms:
            data = waveforms[evid]
            if len(data) < 0.75 * target_event_duration_samp:
                continue
            pad = len(data) - target_event_duration_samp
            if pad < 0:
                data = data[: len(data) + pad]
            elif pad > 0:
                data = np.pad(data, pad)
            data_rsp.append(data)
            valid_evids.append(evid)
    data_rsp = np.asarray(data_rsp)
    # breakpoint()
    data_rsp = resampy.resample(
        data_rsp,
        sampling_rate_accelerated_signal_hz,
        int(audio_sr),
        filter="kaiser_fast",
        parallel=True,
    )
    tend = give_time()
    print(f"{tend-tstart:.2f}sec to resample the waveforms!")

    if scaling_fun == "lin":
        scaling_fun = partial(
            scaling_fun,
            loglin="lin",
            minE=energy_min,
            maxE=energy_max,
        )
    elif scaling_fun == "log":
        scaling_fun = partial(
            scaling_fun,
            loglin="log",
            minE=energy_min,
            maxE=energy_max,
        )
    else:
        print("Custom scaling function")
        scaling_fun = partial(
            scaling_fun,
            scaling_attr_min=energy_min,
            scaling_attr_max=energy_max,
        )

    # for k in tqdm(range(len(catalog)), desc="Concatenating sounds"):
    for k, current_id in enumerate(tqdm(valid_evids, desc="Concatenating sounds")):
        # current_id = catalog.index[k]
        # print(current_id, current_id in waveforms)

        if current_id in waveforms:
            # Test if the data is not going beyond the limits we set for the sound track
            # FIX this so that there is extra time at the end !
            # if movie_starttimes[k] + real_event_dur / speed_factor < movie_duration:
            data = waveforms[current_id]

            if len(data) > 0.75 * (
                real_event_dur * sampling_rate
            ):  # Process only if the data is long enough (entire data should be 20 s @ 500 Hz)
                # RESAMPLE

                data_resampled = data_rsp[k]
                data_norm = normalization(data_resampled)
                if np.isnan(data_norm.max()):
                    breakpoint()

                scale_fac = scaling_fun(catalog.loc[current_id, column_for_scaling])

                data_rescaled = data_norm * scale_fac
                if np.isnan(data_rescaled.max()):
                    breakpoint()

                # ADD TO THE SOUNDTRACK!
                data_db = data_rescaled  # *C/distances[isp]

                i_start = int(movie_starttimes[k] * audio_sr)
                i_end = i_start + len(data_db)

                try:
                    tracks[:, i_start:i_end] = (
                        tracks[:, i_start:i_end] + data_db[None, :]
                    )
                except:
                    print(
                        current_id,
                        k,
                        len(data_db),
                        i_end - i_start,
                        tracks.shape,
                        i_start,
                        i_end,
                    )
                    sys.exit()

            else:
                balbal = 2
        else:
            filenotfoundlist.append(current_id)

    return tracks


def to_db(x, base=20.0):
    """
    Convert a linear amplitude value to decibels (dB).

    Parameters
    ----------
    x : float
        The linear amplitude value to be converted.
    base : float, optional
        The base multiplier for the conversion. The default is 20.0.

    Returns
    -------
    float
        The amplitude value in decibels (dB). Returns negative
        infinity if `x` is less than or equal to 0.
    """
    if x > 0.0:
        return base * np.log10(x)
    else:
        return -np.inf


def scale_fac_lin2(
    scaling_attr,
    scaling_attr_min,
    scaling_attr_max,
    min_sf=0.001,
    max_sf=1.0,
    kink_x=5.0,
    kink_y=0.9,
):
    """
    Compute a piecewise linear scaling factor based on a given attribute.

    Parameters
    ----------
    scaling_attr : float or array-like
        The attribute value to be scaled.
    scaling_attr_min : float
        The minimum value of the attribute.
    scaling_attr_max : float
        The maximum value of the attribute.
    min_sf : float, optional
        The minimum scaling factor. Default is 0.001.
    max_sf : float, optional
        The maximum scaling factor. Default is 1.0.
    kink_x : float, optional
        The x-coordinate of the kink point in the normalized scale. Default is 5.0.
    kink_y : float, optional
        The y-coordinate of the kink point in the normalized scale. Default is 0.9.

    Returns
    -------
    float
        The computed scaling factor.
    """
    scaling_attr = max(scaling_attr, scaling_attr_min)
    norm_scaling_attr = (scaling_attr - scaling_attr_min) / (
        scaling_attr_max - scaling_attr_min
    )
    norm_kink_x = (kink_x - scaling_attr_min) / (scaling_attr_max - scaling_attr_min)
    slope1 = (kink_y - min_sf) / norm_kink_x
    slope2 = (max_sf - kink_y) / (1.0 - norm_kink_x)
    if norm_scaling_attr <= norm_kink_x:
        scale_fac = min_sf + slope1 * norm_scaling_attr
    else:
        scale_fac = kink_y + slope2 * (norm_scaling_attr - norm_kink_x)
    return scale_fac


def scale_fac_calc(loglin, E, minE, maxE, minSF=0.001, maxSF=1):
    """
    Compute a scaling factor based on the given energy level and scaling method.

    Parameters
    ----------
    loglin : {'lin', 'lin2', 'log'}
        The scaling method to use. 'lin' for linear, 'lin2' for normalized linear,
        and 'log' for logarithmic scaling.
    E : float
        The energy level to be scaled.
    minE : float
        The minimum energy level.
    maxE : float
        The maximum energy level.
    minSF : float, optional
        The minimum scaling factor. Default is 0.001.
    maxSF : float, optional
        The maximum scaling factor. Default is 1.0.

    Returns
    -------
    float
        The computed scaling factor.
    """
    E = max(minE, E)
    if loglin == "lin":
        slope = (maxSF - minSF) / (maxE - minE)
        scale_fac = minSF + slope * (E - minE)
    elif loglin == "lin2":
        normalized_E = (E - minE) / (maxE - minE)
        scale_fac = minSF + (maxSF - minSF) * normalized_E
    elif loglin == "log":
        slope = (maxSF - minSF) / (to_db(maxE) - to_db(minE))
        scale_fac = minSF + slope * (to_db(E) - to_db(minE))
    return scale_fac


def make_single_wavesound(
    num_speakers, waveform, sampling_rate, speed_factor, audio_sampling_rate
):
    """
    Generate a multi-speaker waveform by resampling and normalizing the input waveform.

    Parameters
    ----------
    num_speakers : int
        The number of speaker channels to generate.
    waveform : array_like
        The input waveform data.
    sampling_rate : float
        The sampling rate of the input waveform in Hz.
    speed_factor : float
        The factor by which to adjust the speed of the waveform.
    audio_sampling_rate : float
        The desired output sampling rate in Hz.

    Returns
    -------
    tracks : numpy.ndarray
        The generated multi-speaker waveform, with shape (num_speakers, len(data_norm)).
    """
    # num_speakers = 2
    data = waveform

    sampling_rate_accelerated_signal_hz = sampling_rate * speed_factor
    if (
        float(int(sampling_rate_accelerated_signal_hz))
        != sampling_rate_accelerated_signal_hz
    ):
        warnings.warn(
            "Sampling rate * speed_factor cannot be unambigously"
            " interpreted as an integer. It will be rounded down to the"
            " nearest integer."
        )
    sampling_rate_accelerated_signal_hz = int(sampling_rate_accelerated_signal_hz)

    data_resampled = resampy.resample(
        data, sampling_rate_accelerated_signal_hz, int(audio_sampling_rate)
    )

    data_norm = normalization(data_resampled)

    tracks = np.zeros(
        (
            num_speakers,
            len(data_norm),
        ),
        dtype=np.float32,
    )

    tracks[:, :] = data_norm

    return tracks


def round_time(t, sr):
    """
    Parameters
    -----------
    t: scalar float,
        Time, in seconds, to be rounded so that the number
        of meaningful decimals is consistent with the precision
        allowed by the sampling rate.
    sr: scalar float, default to cfg.SAMPLING_RATE_HZ,
        Sampling rate of the data. It is used to
        round the time.

    Returns
    --------
    t: scalar float,
        Rounded time.
    """
    # convert t to samples
    t_samp = np.int64(t * sr)
    # get it back to seconds
    t = np.float64(t_samp) / sr
    return t
