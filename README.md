# README #

## Installation ##

1. Create a virtual environment and ensure Git works in it.
3. Run: `pip install git+https://github.com/aarongerston/XtrUtils/`

## Uninstallation

Run: `pip uninstall XtrUtils`
    
## Xtrodes Offline Signal Statistics and Manipulation class example ##

    from XtrUtils.stats import Stats
    from XtrUtils.utils import Utils
    from XtrUtils.filterbank import Filterer

    # Prepare some data (pay attention to the axes, samples are on axis=0)
    fs = 4000  # Define sampling rate: 4000Hz
    data = np.random.rand(4000, 16)  # Artificial data set
    time = np.arange(0, len(data))/fs  # Artificial timestamp vector

    # Subtract reference channel (in this case, channel 0) from data
    re_referenced_data = Utils.re_reference(data, reref_col=0)

    # Apply custom filters to data
    filters = {'comb': {'W': 50, 'Q': 30},
               'bandpass': {'W': (0.5, 350), 'N': 4}}
    montage1 = Filterer.filter_data(re_referenced_data, filters, fs=fs)

    # Filter channels 0-9 with standard EEG and channels 10-15 with standard EMG filters
    montage2 = Filterer.filter_data(data, Filterer.eeg_filter, fs,
                                    cols=(col for col in range(10)))
    montage2 = Filterer.filter_data(montage2, Filterer.emg_filter, fs,
                                    cols=(10, 11, 12, 13, 14, 15))

    # Calculate (and optionally plot) signal RMS
    rms = Stats.rms(montage1, ts, fs,  # These 3 arguments are required
                    annotations=None,
                    cols=(2, 4, 5),
                    channel_names=('col2', 'col4', 'col5'),
                    window=(10, 444.8),  # Specify signal segment of interest in seconds
                    window_idc=None,  # Specify signal segment of interest in samples
                    rms_type='rolling',
                    roll_window=fs/2,  # Window size (in samples) to calculate rolling RMS
                    plot=True,  # Plot RMS or no?
                    overlay_plots=True,  # Plots RMS on top of input signal
                    plot_trigger=True,  # Only works if annotations are given
                    grid='both',  # Add grid lines on major and minor axes
                    ylim=(-40, 50),  # Specify y-axis limits
                    title='RMS of artificial signal',
                    legend=True
                   )

    # Calculate features
    features = Stats.extract_features_from_window(montage2, ts, fs,  # These 3 arguments are required
                                                  window=None,  # If None, use whole data set
                                                  feature_length=fs*5,
                                                  cols=None,  # Specify specific columns to analyze)
    # Plot features
    Stats.plot_feats(feats,  # This argument is required
                     trig_dict: {0: 'Signal'},  # Also required. Maps trigger values (default=0) to descriptions
                     title: str = 'Montage 2 features',
                     nrows: int = None  # Specify # of rows of plots in figure
                    )

    print('Done')