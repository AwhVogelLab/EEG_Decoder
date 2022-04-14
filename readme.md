# EEG-Decoder

This is a (hopefully) flexible package for decoding epoched EEG data in Python.  It includes support for loading and managing EEG data (in .mat format), processing epoched data, cross validation, classification, and visualization with statistical testing. While most of the functionality is geared towards within-subject, timepoint by timepoint analyses, this package also supports cross-session and cross-subject classification. While this is primarily meant to be useful for members of the Awh/Vogel Lab, we hope that it is general for use by others. This package is currently being developed, use at your own risk! Always check your code, look up functions, and reach out to us if you have questions!

## eeg_decoder.py

### `Experiment`

Organizes and loads in EEG, trial labels, behavior, eyetracking, and session data. 

[TO DO: Function explanations.]

### `Experiment_Syncher`

Synchronizes data between different experiments. Particularly useful for participants who completed multiple sessions or experiments.

### `Wrangler` 

Data processing and cross-validation.

### `Classifier`

Classification and storing of classification outputs.

### `Interpreter`

Visualization and statistical testing.

### `ERP`

Visualization of EEG data.