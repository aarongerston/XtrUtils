import numpy as np
from typing import Union, Optional
import scipy.signal as sig


class Filterer(object):
    
    eeg_filter = {'comb': {'W': 50, 'Q': 30},
                  'bandpass': {'W': (0.5, 35), 'N': 4}}
    emg_filter = {'comb': {'W': 50, 'Q': 30},
                  'bandpass': {'W': (30, 350), 'N': 4}}

    _VALID_FILTERS = ('notch', 'bandpass', 'lowpass', 'highpass', 'comb')

    @staticmethod
    def filter_data(data: np.ndarray,
                    filters: dict,
                    fs: float,
                    cols: Union[tuple, list, None] = None)\
            -> Optional[np.ndarray]:
        """

        :rtype: np.ndarray
        :param data: 2D np.ndarray data set. Rows are samples, columns are channels.
        :param filters: Dictionary of filters. Keys are ('notch', 'bandpass', 'lowpass', 'highpass', 'comb'). Values are
                        themselves dictionaries of filter parameters. For notch and comb filters, keys are 'Q': Q-factor
                        (attenuation strength), and 'W': center frequency. For high/low/bandpass filters, keys are 'W':
                        cutoff frequency (or 2-item list of cutoff frequencies, in the case of bandpass), and 'N':
                        filter order. High/low/bandpass filters are Butterworth.
        :param fs: Sampling rate (Hz).
        :param cols: (OPTIONAL) specifies which columns of <data> to filter. If None, <cols> = data.shape[1].
        :return: 2D np.ndarray of filtered data. Same number of columns as <cols>.
        """

        out = data.copy()

        cols = cols if cols else [col for col in range(data.shape[1])]
        data = data[:, cols]

        if not filters:
            print('Must specify filters.')
            return
        elif not any([k for k in filters.keys() if k in Filterer._VALID_FILTERS]):
            print('FILTER(S) "%s" NOT SUPPORTED. NO FILTERING PERFORMED.' % [k for k in filters.keys()])
            return

        # First interpolate any nans since filtfilt can't handle them
        if np.isnan(data).any():
            nans = np.argwhere(np.isnan(data))
            np.nan_to_num(data)
        else:
            nans = None

        print('Filtering: ')

        for filter, specs in filters.items():
            filter = filter.lower()
            print('\t%s Hz %s' % (str(specs['W']), filter))
            if filter == 'notch':
                if 'Q' not in specs.keys():
                    Q = 60
                else:
                    Q = specs['Q']
                W = specs['W']

                # Create and apply notch filter with center frequency W (Hz)
                b, a = sig.iirnotch(W, Q=Q, fs=fs)
            elif filter == 'comb':
                if 'Q' not in specs.keys():
                    Q = 60
                else:
                    Q = specs['Q']
                W = specs['W']

                # Create comb filter with center frequency W (Hz)
                b, a = sig.iircomb(W, Q, ftype='notch', fs=fs)
            elif filter in ['bandpass', 'lowpass', 'highpass', 'bandstop']:
                if 'N' not in specs.keys():
                    N = 4
                else:
                    N = specs['N']
                W = specs['W']
                if filter == 'bandpass' and W[1] == fs / 2:
                    filter = 'highpass'
                    W = W[0]
                sos = sig.butter(N=N, Wn=W, btype=filter, fs=fs, output='sos')
            else:
                print('FILTER "%s" NOT SUPPORTED.' % filter.upper())
                continue

            for ch in cols:
                if filter in ('bandpass', 'lowpass', 'highpass', 'bandstop'):
                    out[:, ch] = sig.sosfiltfilt(sos, out[:, ch])
                else:
                    out[:, ch] = sig.filtfilt(b, a, out[:, ch])

        if nans is not None:
            out[nans, cols] = np.nan

        return out

    @staticmethod
    def decimate(data: np.ndarray, ts: np.ndarray, fs: float, decimation_factor: int = 1):
        """
        Decimates 2D <data> matrix column-wise by a factor of <decimation_factor>.

        :param data: 2D data matrix. Rows are samples, columns are channels.
        :param ts: 1D timestamp vector equal in samples to number of rows of <data>.
        :param fs: sampling rate.
        :param decimation_factor: returned data is decimated by a factor of <decimation_factor>
        :return: Decimated data matrix, <decimation_factor>-times smaller than the input <data>.
        """

        if (decimation_factor == 1) or any(arg for arg in (data, ts, fs, decimation_factor) is None):
            return data, ts, fs

        # Performs the same computation as
        # ``data = sig.decimate(data, q=decimation_factor, axis=0, n=2)``
        # except that the computation is done column-by-column instead of all at once to prevent crashing
        system = sig.dlti(*sig.cheby1(N=2, rp=0.05, Wn=0.8 / decimation_factor))
        b, a = system.num, system.den
        for col in range(data.shape[1]):
            data[:, col] = sig.filtfilt(b, a, data[:, col], axis=0)
        data = data[::decimation_factor, :]

        # Downsample timestamp vector without filtering
        ts = ts[::decimation_factor]

        # Update effective sampling rate
        fs = fs / decimation_factor

        return data, ts, fs
