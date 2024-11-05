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
    from scipy.signal import butter, lfilter

    b, a = butter(N=N, Wn=fmax, btype="low", fs=sr)

    return lfilter(b, a, data)


# THESE ARE ALL FROM "CMMR_class.py" (Arthur's file from CMMR meeting prep)


def gaussian(mu, sigma, t):
    # x = (1/(np.sqrt(2*np.pi*sigma**2)))*np.exp(-((t-mu)/(np.sqrt(2)*sigma))**2)
    x = np.exp(
        -(((t - mu) / (np.sqrt(2) * sigma)) ** 2)
    )  # unscaled version, but who cares?
    x = x / np.amax(x)
    return x


def phi(alpha, t):
    # center_t = t[int(len(t)/2)]
    # t2 = t-center_t # center t array around 0
    x = 0.5 * (1 + spspec.erf(alpha * t / np.sqrt(2)))
    return x


def EuclDistance(pt1, pt2):
    """Calculate the Euclidian distance between two points defined by their coordinates
    $d = \sqrt{(pt2[0]-pt1[0])^2+(pt2[1]-pt1[1])^2}$ in the 2-d case
    $d = \sqrt{\sum_{i=0}^{n-1}{pt2[i]-pt1[i])^2}}$ in the n-d case

    IN:
        - pt1: a point defined by a list (or array) of coordinates
        - pt2: another point defined by a list (or array) of coordinates
    OUT:
        - dist: the Euclidian distance between these two points
    """
    # n_dims = len(pt1) # number of dimensions
    # sum__ = 0

    # for i in range(n_dims):
    # 	sum__ = sum__ + (pt2[i]-pt1[i])**2
    # dist = np.sqrt(sum__)
    dist = np.sqrt(np.sum((np.asarray(pt1) - np.asarray(pt2)) ** 2))
    return dist


def Coordinates_Polygon(center, n_vertices, radius, angle_shift):
    """Calculate the coordinates of the vertices of a polygon, given a center, a radius, and a number of vertices
    For now it only works in the 2d-case, more dimensions to come!

    IN:
        - center: the coordinates (2-element array or list) of the center of the polygon
        - n_vertices: the number of vertices (e.g. 5 for a pentagon, 6 for a hexagon, & please update your ancient Greek for more information)
        - radius: all vertices will be on a circle of this radius, centered at the center of the polygon
        - angle_shift: used to rotate the circle (angle_shift=0 will place the first point directly above the center), should be in radians

    OUT:
        - x_coords: a list of x-coordinates of the n_vertices vertices
        - y_coords: a list of yx-coordinates of the n_vertices vertices
    """
    x_coords = np.zeros(n_vertices)
    y_coords = np.zeros(n_vertices)

    for k in range(n_vertices):
        x_coords[k] = center[0] + radius * np.sin(
            k * 2 * np.pi / n_vertices + angle_shift
        )
        y_coords[k] = center[1] + radius * np.cos(
            k * 2 * np.pi / n_vertices + angle_shift
        )

    return x_coords, y_coords


def AverageFilter(x, M):
    """M-point moving-average filter on input array x

    IN:
        - x: input array
        - M: width (in samples) of the moving-average window

    OUT:
        - xavg: filtered array
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
    """This function applies a fade in and a fade out of specific duration onto the input signal

    IN:
        - inputsig: the raw signal
        - dur_fadein: the duration of the fade in, in SAMPLES
        - dur_fadeout: the duration of the fade out, in SAMPLES
        - fadetype: the type of fade in and out ("lin" resp. "log" for a linear resp. logarithmic slope between 0 and 1 or 1 and 0)
        - datatype: the type of data in the numpy array Amplitudes (such as np.float32), in order to avoid to increase the size of resulting sounds (default datatype for numpy is float64...)
    OUT:
        - outputsig the signal with fades"""

    Amplitudes = np.ones(len(inputsig), dtype=datatype)

    if fadetype == "lin":
        Amplitudes[0 : int(dur_fadein)] = np.linspace(0, 1, int(dur_fadein))
        Amplitudes[len(inputsig) - int(dur_fadeout) : len(inputsig)] = np.linspace(
            1, 0, num=int(dur_fadeout)
        )
    elif fadetype == "log":
        Amplitudes[0 : int(dur_fadein)] = np.logspace(-100, 0, num=int(dur_fadein))
        Amplitudes[len(inputsig) - int(dur_fadeout) : len(inputsig)] = np.logspace(
            0, -100, num=int(dur_fadeout)
        )

    outputsig = Amplitudes * inputsig

    return outputsig


def linmap(x, in_min, in_max, out_min, out_max):
    """Just a linear mapping of incoming data "x" assumed to range within [in_min:in_max] into range [out_min:out_max]"""
    slope = (out_max - out_min) / (in_max - in_min)
    intercept = out_max - slope * in_max

    mapped_x = slope * x + intercept

    return mapped_x


def normalization(x):
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

    # for isp in range(num_speakers):
    #    exec("track_" + str(isp + 1) + "= np.zeros(int(movie_duration*audio_sr))")

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
                data = data[:len(data) + pad]
            elif pad > 0:
                data = np.pad(data, pad)
            data_rsp.append(data)
            valid_evids.append(evid)
    data_rsp = np.asarray(data_rsp)
    #breakpoint()
    data_rsp = resampy.resample(
            data_rsp,
            sampling_rate_accelerated_signal_hz,
            int(audio_sr),
            filter="kaiser_fast",
            parallel=True
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


    #for k in tqdm(range(len(catalog)), desc="Concatenating sounds"):
    for k, current_id in enumerate(tqdm(valid_evids, desc="Concatenating sounds")):
        #current_id = catalog.index[k]
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

                #data_resampled = resampy.resample(
                #    data, sampling_rate_accelerated_signal_hz, int(audio_sr)
                #)  # resample to audio rate
                data_resampled = data_rsp[k]
                data_norm = normalization(data_resampled)
                if np.isnan(data_norm.max()):
                    breakpoint()

                #scale_fac = scale_fac_calc(
                #    loglin,
                #    catalog.loc[current_id, column_for_scaling],
                #    energy_min,
                #    energy_max,
                #    0.001,
                #    1,
                #)
                scale_fac = scaling_fun(catalog.loc[current_id, column_for_scaling])
                #scale_fac = scale_fac_lin2(
                #    catalog.loc[current_id, column_for_scaling],
                #    energy_min,
                #    energy_max,
                #    0.001,
                #    1,
                #)

                data_rescaled = (
                    data_norm * scale_fac
                )
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
    if x > 0.0:
        return base * np.log10(x)
    else:
        return -np.inf


def scale_fac_lin2(
    scaling_attr, scaling_attr_min, scaling_attr_max, min_sf=0.001, max_sf=1.0,
    kink_x=5.0, kink_y=0.9
):
    scaling_attr = max(scaling_attr, scaling_attr_min)
    norm_scaling_attr = (scaling_attr - scaling_attr_min) / (
        scaling_attr_max - scaling_attr_min
    )
    norm_kink_x = (kink_x - scaling_attr_min) / (
            scaling_attr_max - scaling_attr_min
            )
    slope1 = (kink_y - min_sf) / norm_kink_x
    slope2 = (max_sf - kink_y) / (1. - norm_kink_x)
    if norm_scaling_attr <= norm_kink_x:
        scale_fac = min_sf + slope1 * norm_scaling_attr
    else:
        scale_fac = kink_y + slope2 * (norm_scaling_attr - norm_kink_x)
    return scale_fac


def scale_fac_calc(loglin, E, minE, maxE, minSF=0.001, maxSF=1):
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
