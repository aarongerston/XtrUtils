# External dependencies
import sys
import numpy as np
from typing import Union, Optional
import matplotlib as mpl

# Modifications
mpl.use('Qt5Agg')


class Utils(object):

    @staticmethod
    def re_reference(data: np.ndarray, reref_col: int):

        data = data.copy()

        # Re-reference, if specified
        if type(reref_col) is int:
            data = data - data[:, reref_col]
            data = np.delete(data, reref_col, 1)

        return data

    @staticmethod
    def islist(obj) -> bool:
        """
        Returns true if the object is a list, tuple, or np.ndarray and False otherwise.

        :param obj: object to check
        :return: is the object a list or not?
        """

        return type(obj) in (list, tuple, np.ndarray)

    @staticmethod
    def get_window(ts: Union[np.ndarray],
                   window: Union[list, tuple, np.ndarray] = None,
                   window_idc: Union[list, tuple, np.ndarray] = None)\
            -> tuple:
        """
        Parses <window> and/or <window_idc> to return the desired start and end indices. <ts> is required to determine the
        default end index of len(<ts>).

        :param ts: 1D timestamp vector.
        :param window: 2-element float vector of start and end times in seconds.
        :param window_idc: 2-element int vector of start and end indices.
        :return: [0]: start index
                 [1]: end index
        """

        # Default return values:
        start = 0
        end = len(ts)

        if window is not None and window_idc is not None:
            print('<window> and <window_idc> cannot both be given.')
            return start, end

        if Utils.islist(window_idc):
            start = max(window_idc[0], 0)
            end = min(window_idc[1], len(ts)+start)
        elif Utils.islist(window):
            try:
                start = np.where(ts >= window[0])[0][0]  # error if np.where[0] is []
            except IndexError:
                pass
            try:
                end = np.where(ts < window[1])[0][-1]  # error if np.where[0] is []
            except IndexError:
                pass

        return start, end

    @staticmethod
    def find_breaks(signal: Union[list, tuple, np.ndarray],
                    greater_than: float,
                    search_window: Union[list, tuple, np.ndarray] = None):
        """
        Returns list of indices of 1D array <signal> whose next consecutive sample is at least <greater_than> greater in
        value than the value of the returned index.

        :param signal: 1D array.
        :param greater_than: minimum difference between two consecutive samples that merits returning an index.
        :param search_window: (OPTIONAL) 2-item list of indices that define the search window within <signal>.
        :return:
        """

        if search_window is None:
            search_window = [0, len(signal)]
        else:
            if not Utils.islist(search_window) or len(search_window) != 2:
                print('Error: search_window must be a 2-element list of indices or None.')
                return []

        return np.where(np.diff(signal[search_window[0]:search_window[1]]) > greater_than)[0] + search_window[0]

    @staticmethod
    def get_trig_from_value(value: int, trig_dict: dict):
        """
        Returns trigger text from value, according to trig_dict mapping.

        :param value:
        :param trig_dict:
        :return:
        """

        if trig_dict == {}:
            return None

        if value == 0:
            return None

        return list(trig_dict.keys())[list(trig_dict.values()).index(value)]

    @staticmethod
    def get_trig_vec(annotations,
                     ts: np.ndarray) -> (np.ndarray, dict):

        # Default = all zeros / empty
        trig_vec = np.zeros_like(ts)
        trigs = {}

        # Remove unwanted events
        invalid = ('Change mode', 'File started', 'Calibration Started', 'Calibration Completed',
                   'Recording Started', 'Recording Paused', 'Recording Stopped')
        # onsets = (annotations.onset[n] for n in range(annotations) if annotations.description[n] not in invalid)
        valid_idc = [idx for idx, evt in enumerate(annotations.description) if evt not in invalid]
        # _, valid_idc = np.where(annotations.description not in invalid)
        onsets = annotations.onset[valid_idc]
        events = annotations.description[valid_idc]

        # Remove out-of-bounds events
        valid_idc = [idx for idx, onset in enumerate(onsets) if ts[0] <= onset <= ts[-1]]
        if not valid_idc:
            return trig_vec, trigs
        else:
            first_valid = valid_idc[0]
            last_valid = valid_idc[-1] + 1
        if first_valid > 0:
            first_valid = onsets[first_valid-1]
        onsets = onsets[first_valid:last_valid]
        events = events[first_valid:last_valid]

        # Make trigger vector
        ntrig = 0
        trigs = {}
        for n, (onset, event) in enumerate(zip(onsets, events)):

            # trigger start index = index of ts closest to onset
            on_idx = np.argmin(np.abs(ts - onset))

            # trigger end index = min(end of ts, index of ts closest to next onset - 1)
            if n == len(events) - 1:
                off_idx = len(ts) - 1
            else:
                off_idx = np.argmin(np.abs(ts - onsets[n+1])) - 1

            # map each trigger to a unique value
            if event not in list(trigs.keys()):
                if all([keyword in event for keyword in ('Calibration', 'stopped')]):
                    trigs[event] = 0
                else:
                    ntrig = ntrig+1
                    trigs[event] = ntrig

            # assign the value of the given trigger to the proper trig_vec indices
            trig_vec[on_idx:off_idx] = trigs[event]

        return trig_vec, trigs

    @staticmethod
    def combine_dicts(d1: dict, d2: dict):

        # out = {}
        #
        # for key in set(d1.keys() + d2.keys()):
        #     try:
        #         out.setdefault(key, []).append(d1[key])
        #     except KeyError:
        #         pass
        #
        #     try:
        #         out.setdefault(key, []).append(d2[key])
        #     except KeyError:
        #         pass
        #
        # return out

        out = {}
        for key in set(d1.keys() | d2.keys()):
            try:
                v1 = d1[key]
                if not Utils.islist(v1):
                    v1 = [v1]
            except KeyError:
                v1 = []
            try:
                v2 = d2[key]
                if not Utils.islist(v2):
                    v2 = [v2]
            except KeyError:
                v2 = []
            try:
                out[key] = np.concatenate((v1, v2))
            except:
                print('Unprecedented error in Utils.combine_dicts().')

        # return {
        #     k: np.ravel([d[k] for d in (d1, d2) if k in d])
        #     for k in set(d1.keys()) | set(d2.keys())
        # }
        return out

    def divide_data(data: np.ndarray,
                    trig_vec: np.ndarray,
                    channels: Union[list, tuple] = None,
                    valid_codes: Union[dict, list, tuple] = None,
                    avg: Union[str, None] = 'median',
                    title: str = None,
                    show: bool = True,
                    trig_dict: dict = None,
                    fs: float = None,
                    fmin: float = 0,
                    fmax: float = 80) \
            -> dict:
        """
        Divides data by trigger.

        :param data: 2D np.ndarray data set. Rows are samples, columns are channels.
        :param ts: 1D np.ndarray timestamp vector.
        :param trig_vec: 1D np.ndarray vector of integer values corresponding to unique triggers.
        :param valid_codes: (OPTIONAL) dict or list of triggers to include. Default is all.
        :param avg: (OPTIONAL) if 'mean' or 'median', the resulting dict will reflect data averaged across all channels.
        :param title: (OPTIONAL) text atop figure.
        :param show: (OPRIONAL) show figure or no? One column shows signals where each row is a trigger. Multiple
                     repetitions of a single trigger are overlaid atop each other in the same plot. t=0 represents the start
                     of each trigger segment. The second column shows the corresponding power spectra, with similarly
                     overlaid lines corresponding to each repetition.
        :return: dict of recording segments by trigger. Each key is a key of TRIG_CODES; each value is a list of signal
                 segments with that trigger
        """

        # Average all processed signals into a single mean signal
        # data = average_data(data, avg)

        if show and not trig_dict:
            print('If show==True, you must provide trig_dict.')
            return {}
        if show and not fs:
            print('If show==True, you must provide fs.')
            return {}

        print('Dividing data by triggers:')

        data = np.atleast_2d(data)
        if data.shape[1] > data.shape[0]:
            data = np.transpose(data)

        # trig_vec = data['Trigger']
        # ts = data['Timestamp (seconds)']
        trigs = np.unique(trig_vec)
        trigs = np.append(np.sort(trigs[trigs < 0])[::-1], np.sort(trigs[trigs > 0]))
        # trigs.sort()
        # if 0 in trigs:
        #     trigs = np.delete(trigs, np.where(trigs == 0))
        if valid_codes:
            # trigs = [t for t in trigs if t in valid_codes.keys()]
            if type(valid_codes) is dict:
                trigs = [t for t in trigs if t in [v[0] for v in valid_codes.values()]]
            elif Utils.islist(valid_codes):
                if type(valid_codes[0]) is str:
                    trigs = [C.CODES_ALL[t][0] for t in valid_codes if C.CODES_ALL[t][0] in trigs]
                elif isinstance(valid_codes[0], numbers.Number):
                    trigs = valid_codes
        n_trigs = len(trigs)
        ncols = data.shape[1] if data.ndim == 2 else 1
        channels = channels if channels else [col for col in range(ncols)]
        n_channels = len(channels)
        if n_channels == 0:
            print('Error: data set does not contain channels: ' + str(channels))
            return {}

        # Indices of each trigger onset/offset
        axis = -1 if ((trig_vec.ndim == 1) or (trig_vec.shape[1] > trig_vec.shape[0])) else 0
        trigs_offsets = np.nonzero(np.diff(trig_vec, axis=axis))[0]
        # trigs_offsets = [trig for trig in trigs_offsets if trig-1 not in np.insert(trigs_offsets, 0, 0)]
        trigs_onsets = [trig + 1 for trig in trigs_offsets]
        trigs_onsets = np.insert(trigs_onsets, 0, 0)
        trigs_offsets = np.insert(trigs_offsets, len(trigs_offsets), len(trig_vec) - 1)

        if show:
            import matplotlib.pyplot as plt
            from XtrViz.plotter import Plotter
            f, ax = plt.subplots(nrows=n_trigs, ncols=2, tight_layout={'pad': 2.3, 'h_pad': 0.3})
            ax = np.atleast_2d(ax)
            Plotter.maximize_fig()
            f.suptitle(title)

        # Iterate over the different triggers that exist in this recording
        segments = {}
        for n_trig, trig_val in enumerate(trigs):

            # trig_str = Utils.get_trig_from_value(trig_val, trig_dict)
            # sys.stdout.write(f'\r\t{n_trig + 1}/{n_trigs}: {trig_str}')
            # sys.stdout.flush()
            sys.stdout.write(f'\r\t{n_trig + 1}/{n_trigs}')

            trig_idc = np.where(trig_vec == trig_val)[0]

            # Onset and offset indices of the given trigger
            trig_onsets = np.sort(tuple(set(trigs_onsets).intersection(trig_idc)))
            trig_offsets = np.sort(tuple(set(trigs_offsets).intersection(trig_idc)))
            if trig_onsets[0] > trig_offsets[0]:
                trig_offsets = trig_offsets[1:]
            if trig_onsets[-1] > trig_offsets[-1]:
                trig_onsets = trig_onsets[:-1]
            if len(trig_onsets) != len(trig_offsets):
                print('Something went wrong here...')

            trig_segments = []
            for n_segment, onset in enumerate(trig_onsets):
                # d_idc = np.arange(trig_onsets[n_segment], trig_offsets[n_segment])
                d_segment = data[trig_onsets[n_segment]:trig_offsets[n_segment], :]
                if not np.any(d_segment):
                    continue
                trig_segments.append(d_segment)

                # Plot
                if show:
                    color = Plotter.get_trig_color(trig_val, trig_dict)
                    seg = d_segment[:, channels]
                    seg_ts = np.arange(len(seg)) / fs
                    ax[n_trig, 0].plot(seg_ts, seg, color=color, alpha=0.5 / n_channels)

                    # Plot FFT
                    for col in channels:
                        Plotter.plot_spectrum(d_segment[:, col], seg_ts, fs, fmin=fmin, fmax=fmax, ax=ax[n_trig, 1], color=color,
                                               alpha=0.5/n_channels, linewidth=2, ylab=True, xlab=(n_trig == n_trigs-1))

            if show:
                # Plot adjustments
                ax[n_trig, 0].text(0.5, 0.8, trig_dict[trig_val],
                                   horizontalalignment='center', transform=ax[n_trig, 0].transAxes)
                ax[n_trig, 1].text(0.5, 0.8, trig_dict[trig_val],
                                   horizontalalignment='center', transform=ax[n_trig, 1].transAxes)
                ax[n_trig, 0].set_ylabel('uV')
                # ax[n_trig, 1].set_ylabel(ylab)
                if n_trig != n_trigs - 1:
                    ax[n_trig, 0].set_xticklabels([])
                    ax[n_trig, 1].set_xticklabels([])

            # Add signal segments to segments dict
            segments[trig_val] = trig_segments

        if show:
            # Add x-labels to bottom plots
            ax[n_trigs - 1, 0].set_xlabel('Time (s)')
            ax[n_trigs - 1, 1].set_xlabel('Frequency (Hz)')

            # Link axes
            Plotter.link_xy(ax[:, 0], triggers=False)
            Plotter.link_xy(ax[:, 1], triggers=False)

        sys.stdout.write(f'\r\t{n_trigs}/{n_trigs} complete.\n')

        return segments