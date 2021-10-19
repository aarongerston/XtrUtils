import sys
import numpy as np
import scipy.stats
import seaborn as sns
import ptitprince as pt
from typing import Union
import scipy.signal as sig
import matplotlib.pyplot as plt

from DataObj import DataObj
from XtrUtils.utils import Utils
from XtrViz.plotter import Plotter


class Stats(object):

    @staticmethod
    def rms(data: np.ndarray,
            ts: np.ndarray,
            fs: float,
            annotations=None,
            channel_names: Union[tuple, list] = None,
            cols: Union[list, tuple, np.ndarray] = None,
            window: Union[list, tuple, np.ndarray] = None,
            window_idc: Union[list, tuple, np.ndarray] = None,
            rms_type: str = 'rolling',
            roll_window: int = 2000,
            plot: bool = True,
            overlay_plots: bool = True,
            plot_trigger: bool = True,
            grid: str = None,
            ylim: Union[list, tuple] = None,
            title: str = None,
            legend: bool = True)\
            -> np.ndarray:
        """
        Returns the rolling RMS (signal equal in length to input signal; <rms_type>='rolling') or static RMS (single value;
        <rms_type>='static') of each data channel in <data> (or the channels specified in <channels>) and optionally plots
        the RMS signal (if <plot> is True), either alone or overlaid over the original signal(s), depending on the value of
        <overlay_plots>. If one of <window> or <window_idc> is specified, rolling RMS only of that window will be calculated
        (and optionally plotted).

        :param data: np.ndarray. Columns are channels, rows are samples.
        :param ts: np.ndarray timestamp vector. Same number of rows as <data>.
        :param fs: sampling rate (Hz).
        :param annotations: mne.annotations.Annotations list of triggers and times.
        :param channel_names: names of columns. Must be equal in length to <cols>, if given, or number of columns in
                              <data> otherwise.
        :param cols: list of columns to analyze. List must contain only ints.
        :param window: 2-item list of floats; start and end time to analyze, in seconds.
        :param window_idc: 2-item list of ints; start and end indices to analyze.
        :param rms_type: 'rolling' or 'static'.
        :param roll_window: window size (in samples) of rolling RMS window. Ignored if <rms>='static'.
        :param plot: (OPTIONAL) plot rolling RMS signals or no? Default: True.
        :param overlay_plots: (OPTIONAL) overlay RMS plots over original signals or no? Default: True.
        :param plot_trigger: (OPTIONAL) add trigger plot beneath RMS plots? Default: True.
        :param grid: (OPTIONAL) y-axis grid lines to add to plots: ['major', 'minor', 'both', None]
        :param ylim: (OPTIONAL) y-axis limits. If None, automatically calculates optimal limits.
        :param title: (OPTIONAL) plot title
        :param legend: (OPTIONAL) show legend at bottom or no? Default: True.
        :return: np.ndarray where each column corresponds to a channel whose RMS was calculated
        """

        # Crop if desired
        start, end = Utils.get_window(ts, window, window_idc)
        ts = ts[start:end]
        data = data[start:end, :]

        # Get relevant stuff
        channels = cols if cols else [col for col in range(data.shape[1])]
        channel_names = channel_names if channel_names else [f'Column {col}' for col in channels]
        title = title if title is not None else ''

        print(f'Calculating RMS of {title} data.')

        # Calculate RMS and store in 2D numpy array same size as data (after cropping)
        rms = data.copy()
        for ch in channels:
            signal = data[:, ch]
            if rms_type == 'rolling':
                rms[:, ch] = Stats.rolling_rms(signal, roll_window)
            elif rms_type == 'static':
                rms[:, ch] = np.sqrt(np.mean(signal**2))

        if plot & (rms_type == 'rolling'):

            if overlay_plots:
                # Plot input signals
                Plotter.plot_signals(ts, data, fs, channels, annotations, channel_names, title, window_idc=None)
                # Fetch axes from signals plots and overlay rms signals over each channel
                ax = plt.gcf().axes
                for n, ch in enumerate(channels):
                    ax[n].plot(ts, rms[:, ch], label='RMS')
                    # Add grid lines
                    if grid:
                        ax[n].grid(which='both', axis='y')
            else:
                # Use plot_signals() with RMS DataFrame to plot RMS
                Plotter.plot_signals(ts, rms, fs, channels, annotations, channel_names, title, window_idc=None)
                ax = plt.gcf().axes
                # Add grid lines
                if grid:
                    for a in ax:
                        a.grid(which='both', axis='y')

            # Add legend
            if legend:
                handles, labels = ax[0].get_legend_handles_labels()
                Plotter.add_trig_legend(plt.gcf(), labels=['Signal', 'RMS'], handles=handles, y_pos=0.02)

            # Axis adjustments
            Plotter.link_xy(ax, triggers=plot_trigger)
            if Utils.islist(ylim) and (len(ylim) == 2):
                ax[0].set_ylim(ylim)

        return rms

    @staticmethod
    def rolling_rms(signal: np.ndarray,
                    n: int)\
            -> np.ndarray:
        """
        Calculates rolling RMS of <signal> using moving windows <N> samples in length.

        :param signal: 1D input array.
        :param n: length (in samples) of moving window for rolling RMS calculation
        :return: rolling RMS signal, equal in length to <signal>.
        """

        sig2 = np.power(signal, 2)
        window = np.ones(int(n))/float(n)
        return np.sqrt(np.convolve(sig2, window, 'same'))

    @staticmethod
    def extract_features_from_window(data: np.ndarray,
                                     ts: np.ndarray,
                                     fs: float,
                                     window: Union[tuple, list, None] = None,
                                     feature_length: Union[int, None] = None,
                                     cols: Union[list, tuple, str, int, None] = None,
                                     blinks: Union[np.ndarray, None] = None) \
            -> dict:

        start, end = Utils.get_window(ts, window)
        data = data[start:end, cols]

        feats = {}
        for ncol, col in enumerate(cols):
            signal = data[:, col]
            feats[ncol] = Stats.calc_features(signal, feature_length, fs, trig=0, blinks=blinks)

        return feats


    @staticmethod
    def calc_features(signal: np.ndarray,
                      feature_length: int,
                      fs: float,
                      trig: int = 0,
                      blinks: Union[np.ndarray, None] = None,
                      extract_only: Union[tuple, list, np.ndarray] = None)\
            -> dict:
        """
        Calculates features from a given 1D signal
        :param signal: 1D signal
        :param feature_length: number of samples for each feature calculation window
        :param fs: recording sampling rate
        :param trig: trigger during signal segment
        :param blinks: (OPTIONAL) blink indices
        :return: dict. Keys are feature names/descriptions, values are corresponding list of feature values. Each data
                 point is calculated within a window of size feature_length, so that there are a total of
                 floor(len(signal)/feature_length) data points calculated for each feature.
        """

        feature_length = int(feature_length)
        feats = ('SD',
                 'MAD',
                 'Power SD',
                 'Power MAD',
                 'Power MAD (standardized)',
                 'Dominant frequency',
                 'Max. power',
                 'Power < 5Hz (low)',
                 'Power ~7.813Hz',
                 'Power ~7.813Hz (rel.)',
                 'Power ~50Hz',
                 'Power ~50Hz (rel.)',
                 'Power > 1000 Hz (high)',
                 'Power 5-400 Hz (med)',
                 'Power high:low',
                 'Power high:med',
                 'Power med:low',
                 'Med. power',
                 'Median peak height',
                 'Highest pos. peak',
                 'Lowest neg. peak',
                 'Most prom. pos. peak prom.',
                 'Most prom. neg. peak prom.',
                 'Mean pos. peak prom.',
                 'Mean neg. peak prom.',
                 'Most prom. pos. peak height',
                 'Most prom. neg. peak height',
                 'Mean dyn. range',
                 'Zero-crossings',
                 'Power slope')
        if Utils.islist(extract_only):
            feats = tuple(set(extract_only).intersection(set(feats)))
        feats = {feat: [] for feat in feats}
        feats['Trigger'] = []
        if blinks is not None:
            feats['Avg. blinks/sec'] = []

        # step = 0.05
        # fmax = 80
        # f = np.linspace(start=step, stop=fmax, num=int(fmax / step), endpoint=True)
        # fmin_delta = step
        # fmax_delta = 4
        # f_delta = np.linspace(start=fmin_delta, stop=fmax_delta, num=int(fmax_delta / step))
        # fmin_theta = 4
        # fmax_theta = 8
        # f_theta = np.linspace(start=fmin_theta + step, stop=fmax_theta, num=int(fmax_theta / step))
        # fmin_alpha = 8
        # fmax_alpha = 14
        # f_alpha = np.linspace(start=fmin_alpha + step, stop=fmax_alpha, num=int(fmax_alpha / step))
        # fmin_beta = 14
        # fmax_beta = 30
        # f_beta = np.linspace(start=fmin_beta + step, stop=fmax_beta, num=int(fmax_beta / step))
        # fmin_gamma = 30
        # fmax_gamma = fmax
        # f_gamma = np.linspace(start=fmin_gamma + step, stop=fmax_gamma, num=int(fmax_gamma / step))

        for n, idx in enumerate(np.arange(len(signal))[::feature_length]):  # signal.index[::feature_length]):
            feat_segment = signal[idx:idx + feature_length]
            if blinks is not None:
                blink_segment = blinks[idx:idx + feature_length]

            # Find nans
            non_nans = ~np.isnan(feat_segment)

            # if entire segment contains < 1 second of data, skip
            if sum(non_nans) < fs:
                continue

            ts_feat = np.arange(len(feat_segment)) / fs

            # # Mean power in each EEG band
            # pxx_d = sig.lombscargle(ts_feat[non_nans], feat_segment[non_nans], 2 * np.pi * f_delta)
            # pxx_t = sig.lombscargle(ts_feat[non_nans], feat_segment[non_nans], 2 * np.pi * f_theta)
            # pxx_a = sig.lombscargle(ts_feat[non_nans], feat_segment[non_nans], 2 * np.pi * f_alpha)
            # pxx_b = sig.lombscargle(ts_feat[non_nans], feat_segment[non_nans], 2 * np.pi * f_beta)
            # pxx_g = sig.lombscargle(ts_feat[non_nans], feat_segment[non_nans], 2 * np.pi * f_gamma)
            # power_d = sum(pxx_d) / len(feat_segment)
            # power_t = sum(pxx_t) / len(feat_segment)
            # power_a = sum(pxx_a) / len(feat_segment)
            # power_b = sum(pxx_b) / len(feat_segment)
            # power_g = sum(pxx_g) / len(feat_segment)
            # 
            # # dominant freq
            # pxx = sig.lombscargle(ts_feat[non_nans], feat_segment[non_nans], 2 * np.pi * f)
            # dom_freq = f[np.argmax(pxx)]

            ## NEW:
            # if any(np.invert(non_nans)):
            #    # lombscargle
            # else:
            pxx = np.abs(np.fft.rfft(feat_segment, axis=-1))
            f = np.fft.rfftfreq(len(feat_segment), 1 / fs)

            # Standardized power spectrum
            pxx_std = (pxx - np.mean(pxx)) / np.std(pxx)

            # total power:
            total_power = np.sum(pxx)

            # average power
            avg_power = np.median(pxx)

            # power < 5Hz
            f5 = np.argmin(np.abs(f - 5))
            med_f5 = np.median(pxx[:f5])

            # freq 7.813Hz
            f7pt813 = np.argmin(np.abs(f - 7.813))
            med_f7pt813 = pxx[f7pt813]

            # freq = 50Hz
            f50 = np.argmin(np.abs(f - 50))
            med_f50 = pxx[f50]

            # power > 1000 Hz
            f1000 = np.argmin(np.abs(f - 1000))
            med_f1000 = np.median(pxx[f1000:])
            
            # power in physiological range
            f400 = np.argmin(np.abs(f - 400))
            med_phys = np.median(pxx[f5:f400])

            # # relative power in each EEG band
            # perc_d = sum(pxx[np.where((f >= fmin_delta) & (f <= fmax_delta))[0]]) / sum(pxx) * 100
            # perc_t = sum(pxx[np.where((f >= fmin_theta) & (f <= fmax_theta))[0]]) / sum(pxx) * 100
            # perc_a = sum(pxx[np.where((f >= fmin_alpha) & (f <= fmax_alpha))[0]]) / sum(pxx) * 100
            # perc_b = sum(pxx[np.where((f >= fmin_beta) & (f <= fmax_beta))[0]]) / sum(pxx) * 100
            # perc_g = sum(pxx[np.where((f >= fmin_gamma) & (f <= fmax_gamma))[0]]) / sum(pxx) * 100

            # Fundamental frequency (from autocorr)
            # try:
            #     f0 = Stats.calc_f0(feat_segment, fs)
            # except IndexError:
            #     f0 = np.nan

            # median height of peaks
            pos_pks, _ = sig.find_peaks(feat_segment)
            neg_pks, _ = sig.find_peaks(-feat_segment)
            pos_proms = sig.peak_prominences([x for x in feat_segment], pos_pks)[0]
            neg_proms = sig.peak_prominences([-x for x in feat_segment], neg_pks)[0]
            pos_pk_heights = [feat_segment[idx] for idx in list(pos_pks)]
            neg_pk_heights = [feat_segment[idx] for idx in list(neg_pks)]

            # Power standard deviation
            if 'Power SD' in feats.keys():
                power_sd = np.std(pxx)
                feats['Power SD'].append(power_sd)

            if 'Dominant frequency' in feats.keys():
                dom_freq = f[np.argmax(pxx)]
                feats['Dominant frequency'].append(dom_freq)

            if 'Max. power' in feats.keys():
                max_power = np.max(pxx)
                feats['Max. power'].append(max_power)

            if 'Power < 5Hz (low)' in feats.keys():
                feats['Power < 5Hz (low)'].append(med_f5)

            if 'Power ~7.813Hz' in feats.keys():
                feats['Power ~7.813Hz'].append(med_f7pt813)

            if 'Power ~7.813Hz (rel.)' in feats.keys():
                feats['Power ~7.813Hz (rel.)'].append(med_f7pt813 / total_power)

            if 'Power ~50Hz' in feats.keys():
                feats['Power ~50Hz'].append(med_f50)

            if 'Power ~50Hz (rel.)' in feats.keys():
                f50_rel = med_f50 / total_power
                feats['Power ~50Hz (rel.)'].append(f50_rel)

            if 'Power > 1000 Hz (high)' in feats.keys():
                feats['Power > 1000 Hz (high)'].append(med_f1000)

            if 'Power 5-400 Hz (med)' in feats.keys():
                feats['Power 5-400 Hz (med)'].append(med_phys)

            if 'Power high:low' in feats.keys():
                hi_low = med_f1000 / med_f5
                feats['Power high:low'].append(hi_low)

            if 'Power high:med' in feats.keys():
                hi_med = med_f1000 / med_phys
                feats['Power high:med'].append(hi_med)

            if 'Power med:low' in feats.keys():
                med_low = med_phys / med_f5
                feats['Power med:low'].append(med_low)

            if 'Med. power' in feats.keys():
                feats['Med. power'].append(avg_power)

            # Median absolute deviation of power spectrum
            if 'Power MAD' in feats.keys():
                power_mad = np.median(np.abs(pxx - avg_power))
                feats['Power MAD'].append(power_mad)

            # Median absolute deviation of standardized power spectrum
            if 'Power MAD (standardized)' in feats.keys():
                power_mad_std = np.median(np.abs(pxx_std - np.median(pxx_std)))
                feats['Power MAD (standardized)'].append(power_mad_std)

            if 'Median peak height' in feats.keys():
                med_pk = np.median(np.abs(np.concatenate([pos_pk_heights, neg_pk_heights]).flatten()))
                feats['Median peak height'].append(med_pk)

            # highest pos. peak
            if 'Highest pos. peak' in feats.keys():
                max_pos_pk = max(pos_pk_heights) if any(pos_pks) else np.nan
                feats['Highest pos. peak'].append(max_pos_pk)

            # most negative neg. peak
            if 'Lowest neg. peak' in feats.keys():
                min_neg_pk = min(neg_pk_heights) if any(neg_pks) else np.nan
                feats['Lowest neg. peak'].append(min_neg_pk)

            # height of most prominent peak
            if 'Most prom. pos. peak height' in feats.keys():
                max_pos_prom_height = pos_pk_heights[int(np.argmax(pos_proms))] if any(pos_pks) else np.nan
                feats['Most prom. pos. peak height'].append(max_pos_prom_height)

            # height of most prominent negative peak
            if 'Most prom. neg. peak height' in feats.keys():
                max_neg_prom_height = neg_pk_heights[int(np.argmax(neg_proms))] if any(neg_pks) else np.nan
                feats['Most prom. neg. peak height'].append(max_neg_prom_height)

            # prominence of most prominent pos. peak
            if 'Most prom. pos. peak prom.' in feats.keys():
                max_pos_prom = max(pos_proms) if any(pos_pks) else np.nan
                feats['Most prom. pos. peak prom.'].append(max_pos_prom)

            # prominence of most prominent negative peak
            if 'Most prom. neg. peak prom.' in feats.keys():
                max_neg_prom = max(neg_proms) if any(neg_pks) else np.nan
                feats['Most prom. neg. peak prom.'].append(max_neg_prom)

            # mean pos. peak prominence
            if 'Mean pos. peak prom.' in feats.keys():
                mean_pos_prom = np.mean(pos_proms) if any(pos_pks) else np.nan
                feats['Mean pos. peak prom.'].append(mean_pos_prom)

            # mean neg. peak prominence
            if 'Mean neg. peak prom.' in feats.keys():
                mean_neg_prom = np.mean(neg_proms) if any(neg_pks) else np.nan
                feats['Mean neg. peak prom.'].append(mean_neg_prom)

            if 'Mean dyn. range' in feats.keys():
                mean_dynamic_range = np.nan
                if any(pos_pks) and any(neg_pks):
                    all_pks = np.sort(np.hstack([pos_pks, neg_pks]))
                    mean_dynamic_range = np.mean(np.abs(np.diff(all_pks)))
                feats['Mean dyn. range'].append(mean_dynamic_range)

            if 'Zero-crossings' in feats.keys():
                d = np.diff(np.sign(feat_segment))
                d[np.isnan(d)] = 0
                zc = float(len(np.where(d)[0]))
                feats['Zero-crossings'].append(zc)

            if 'Power slope' in feats.keys():
                power_slope = scipy.stats.linregress(f, pxx).slope
                feats['Power slope'].append(power_slope)

            if 'MAD' in feats.keys():
                MAD = np.median(np.abs(feat_segment - np.median(feat_segment)))
                feats['MAD'].append(MAD)

            if 'SD' in feats.keys():
                feats['SD'].append(np.std(feat_segment))

            if 'Avg. blinks/sec' in feats.keys():
                n_blinks = sum(blink_segment) / (sum(non_nans) / fs)
                feats['Avg. blinks/sec'].append(n_blinks)

            feats['Trigger'].append(trig)

        return feats

    @staticmethod
    def calc_f0(sig_in, fs) -> float:
        """
        From: https://gist.github.com/endolith/255291
        Calculates fundamental frequency.

        :param sig_in: 1D signal
        :param C: Constants.
        :return: (float) fundamental frequency.
        """

        if all(x == np.nan for x in sig_in):
            return

        # Use only longest non-nan stretch of signal
        signal = max(np.split(sig_in, np.where(np.isnan(sig_in))[0]), key=len)[1:]

        # Calculate autocorrelation and throw away the negative lags
        corr = np.correlate(signal, signal, mode='full')
        corr = corr[len(corr) // 2:]

        # Find the first low point
        d = np.diff(corr)
        start = np.nonzero(d > 0)[0][0]

        # Find the next peak after the low point (other than 0 lag).  This bit is not reliable for long signals, due to the
        # desired peak occurring between samples, and other peaks appearing higher. Should use a weighting function to
        # de-emphasize the peaks at longer lags.
        peak = np.argmax(corr[start:]) + start
        px, py = Stats.parabolic(corr, peak)

        return fs / px

    @staticmethod
    def parabolic(f, x):
        """
        From: https://gist.github.com/endolith/255291

        Quadratic interpolation for estimating the true position of an inter-sample maximum when nearby samples are known.
        f is a vector and x is an index for that vector. Returns (vx, vy), the coordinates of the vertex of a parabola that
        goes through point x and its two neighbors.

        (Used for calculation of fundamental frequency. See calc_f0().)

        Example:
        Defining a vector f with a local maximum at index 3 (= 6), find local maximum if points 2, 3, and 4 actually defined
        a parabola.

        In [3]: f = [2, 3, 1, 6, 4, 2, 3, 1]
        In [4]: parabolic(f, argmax(f))
        Out[4]: (3.2142857142857144, 6.1607142857142856)
        """

        xv = 1 / 2. * (f[x - 1] - f[x + 1]) / (f[x - 1] - 2 * f[x] + f[x + 1]) + x
        yv = f[x] - 1 / 4. * (f[x - 1] - f[x + 1]) * (xv - x)
        return xv, yv

    @staticmethod
    def extract_features_from_triggers(segments: dict,
                                       fs: float,
                                       feature_length: Union[int, None] = None,
                                       cols: Union[list, tuple, str, int, None] = None,
                                       blinks: Union[np.ndarray, None] = None,
                                       extract_only: Union[tuple, list, np.ndarray] = None)\
            -> dict:
        """
        Extracts various features independently for each segment in <segments>. Features are calculated for each
        <feature_length> number of samples per segment.

        :param segments: (dict) (output from basics.divide_data())
        :param fs: sampling rate of system
        :param feature_length: number of samples for each feature calculation window. Defaults to fs (1 second).
        :return: dict with the following keys:
            ('Mean delta power [0-4Hz]',
             'Mean theta power [4-8Hz]',
             'Mean alpha power [8-14Hz]',
             'Mean beta power [14-30Hz]',
             'Mean gamma power [30-80Hz]',
             '% delta power [0-4Hz]',
             '% theta power [4-8Hz]',
             '% alpha power [8-14Hz]',
             '% beta power [14-30Hz]',
             '% gamma power [30-80Hz]',
             'Dominant frequency',
             'Fundamental frequency',
             'Avg. power',
             'Median peak height',
             'Highest pos. peak',
             'Lowest neg. peak',
             'Most prom. pos. peak prom.',
             'Most prom. neg. peak prom.',
             'Mean pos. peak prom.',
             'Mean neg. peak prom.',
             'Most prom. pos. peak height',
             'Most prom. neg. peak height',
             'Mean dyn. range',
             'Zero-crossings',
             'Trigger')
        """

        # If <cols> not given, use all channels of dataset
        ncols = segments[list(segments.keys())[0]][0].shape[1] if segments[list(segments.keys())[0]][0].ndim == 2 else 1
        cols = cols if cols else [col for col in range(ncols)]

        # Recursively call this function separately for each channel desired
        # if Utils.islist(cols):
        #     [Stats.extract_features(segments, fs, feature_length, col, blinks) for col in cols]
        #     [Stats.extract_features_from_triggers(segments, fs, feature_length, col, blinks) for col in cols]
        if type(cols) is int:
            # col = cols
            cols = [cols]

        # Default <feature_length> = fs. <feature_length> must be >= fs for calculation of certain features.
        if not feature_length:
            feature_length = fs
        elif feature_length < fs:
            print('Error in extract_features(): <feature_length> must be >= sampling rate.')
            return {}
        feature_length = int(feature_length)

        print('Extracting features...')

        trigs = segments.keys()

        feats_all = {}
        for col in cols:

            feats = {}
            for n_trig, trig in enumerate(trigs):

                sys.stdout.write(f'\r\tColumn {col+1}/{len(cols)}, Trigger {n_trig + 1}/{len(trigs)}')
                sys.stdout.flush()

                trig_segments = segments[trig]
                for data_segment in trig_segments:
                    signal = data_segment[:, col] if data_segment.ndim == 2 else data_segment

                    segment_feats = Stats.calc_features(signal, feature_length, fs, trig, blinks, extract_only=extract_only)
                    feats = Utils.combine_dicts(feats, segment_feats)

            sys.stdout.write(f'\r\r\t{len(trigs)}/{len(trigs)} complete.\n')

            feats_all[f'Channel {col}'] = feats

        return feats_all

    def plot_feats(feats: dict,
                   trig_dict: dict,
                   title: str = None,
                   nrows: int = None,
                   save_fig: Union[bool, str] = False) \
            -> plt.Figure:
        """
        Plots features as subplots on a single figure.
    
        :param feats: pd.DataFrame where each column is a feature.
        :param title: (OPTIONAL) text displayed atop the figure.
        :param nrows: (OPTIONAL) number of rows to plot. Default = floor(np.sqrt(len(feats))).
        :param save_fig: (OPTIONAL) save the figure or no? If <save_fig> is a str, figure will be saved to the absolute file
                         location specified by the <save_fig> str. If it is True, figure will be saved to C.SAVE_DIRECTORY.
        :return: plt.Figure.
        """

        figs = []
        for set_name, feature_set in feats.items():
            feat_names = [k for k in list(feature_set.keys()) if k != 'Trigger']
            feature_set['Trigger'] = [trig_dict[t] for t in feature_set['Trigger']]
        
            n_feats = len(feat_names)
            # trigs = feats['Trigger']
            # trigs = [np.unique(feature_set['Trigger']) for k in feats.keys()]
            n_trigs = len(np.unique(feature_set['Trigger']))

            print(f"Plotting features for {feat_names}.")
        
            # Prepare figure
            if nrows is None:
                nrows = int(np.sqrt(n_feats))  # 4
            ncols = int(np.ceil(n_feats / nrows))  # 5
            f, ax = plt.subplots(nrows=nrows, ncols=ncols)
            f.subplots_adjust(bottom=0.12, top=0.92, right=0.95, left=0.05, wspace=0.5)
            Plotter.maximize_fig()
        
            # delete extra axes
            n_f = n_feats + 1
            while n_f <= nrows * ncols:
                col = int(np.ceil((n_f - 1) / nrows) - 1)
                row = int((n_f - 1) % col)
                f.delaxes(ax[row, col])
                n_f = n_f + 1
        
            ordered_trigs = [t for t in trig_dict.values() if t in np.unique(np.unique(feature_set['Trigger']))]
            palette = sns.color_palette([Plotter.get_trig_color(t, trig_dict) for t in ordered_trigs])
            
            for n_feat, feat_name in enumerate(feat_names):
        
                nrow = n_feat % nrows
                ncol = int(n_feat / nrows)
        
                # 'hue' parameter shows legend
                if n_trigs > 1:
                    hue = 'Trigger'
                    hue_order = ordered_trigs
                    # hue_order = [Utils.get_trig_from_value(t, trig_dict) for t in ordered_trigs]
                else:
                    hue = None
                    hue_order = None
                rc = pt.RainCloud(x='Trigger', y=feat_name, data=feature_set, ax=ax[nrow, ncol],
                                  hue=hue, hue_order=hue_order, palette=palette, alpha=0.5)  # palette = 'bright'
                # rc = raincloud(x_column_name='Trigger', y_column_name=feat_name, data=feats, C=C, ax=ax[nrow, ncol], alpha=0.5)
        
                # Remove x labels
                rc.set_xlabel('')
                rc.set_xticklabels('')

                rc.set_ylabel(feat_name)
        
                # Get legend data
                handles, labels = rc.get_legend_handles_labels()
                handles = handles[:len(ordered_trigs)]
                labels = labels[:len(ordered_trigs)]
    
            # Remove auto-created legends
            for nrow in range(nrows):
                for ncol in range(ncols):
                    ax[nrow, ncol].legend().remove()
        
            # Add legend
            if labels:
                Plotter.add_trig_legend(f, labels=labels, handles=handles[:n_trigs])
        
            # if title is not None:
            title = (title + '\n' + set_name if title else set_name) if len(feats) > 1 else title
            plt.suptitle(title)
        
            # # save fig
            # if save_fig:
            #     fig_title = title if title else "Features"
            #     fig_title = (lambda x: x.replace('\n', '').replace('=', '').replace('/', '-').replace(' ', '_'))(fig_title)
            #     if type(save_fig) is str:
            #         savedir = save_fig
            #         if not os.path.exists(savedir):
            #             os.makedirs(savedir)
            #         saving.savefig(f, fname=os.path.join(savedir, fig_title))
            #
            #     else:
            #         savedir = os.path.join(C.DATA_DIRECTORY, 'figures')
            #         if not os.path.exists(savedir):
            #             os.makedirs(savedir)
            #         saving.savefig(f, fname=os.path.join(savedir, fig_title))
            
            figs.append(f)
            
        return figs