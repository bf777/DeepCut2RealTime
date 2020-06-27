# DeepCut2RealTime
Welcome to DeepCut2RealTime, an add-on for [DeepLabCut](https://github.com/DeepLabCut/DeepLabCut) that enables real-time tracking
and reinforcement of animal behaviours. This code was used to carry out the behavioural experiments outlined
in our _eNeuro_ paper: [Forys, Xiao, Gupta, and Murphy (2020)](https://doi.org/10.1523/ENEURO.0096-20.2020), 
and builds upon the code outlined in our _bioRxiV_ preprint: [Forys, Xiao, Gupta, Boyd, and Murphy (2018)](https://doi.org/10.1101/482349). 
The included MATLAB files were used to analyze the occurrence of LED flashes during the experiment (for more details, 
see our _eNeuro_ paper).

**NEW:** Test our real-time tracking system on your own computer (no installation required) using Google Colab! Click
the following button to get started:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bf777/deepcut2realtime/blob/master/notebooks/deepcut2realtime.ipynb)

## Features:
- Integrates with DeepLabCut 2.0.6 and later through an easy-to-use iPython interface.
- Combines real-time application of DeepLabCut's pose estimation framework at high
framerates (> 70 Hz) with customizable selection of behaviours to reinforce.

## Important notes:
- At this time, DeepCut2RealTime has only been fully tested with the Sentech STC-MCCM401U3V USB-3 Vision camera; we are
working on making it compatible with standard OpenCV cameras as well.
- If you are not using a high speed camera with a GPU setup, do not expect to get the same framerate as we did in the paper. Typical
frame rates with a MacBook Pro (CPU-based DeepLabCut) with a standard webcam are 15-20 Hz.
- DeepCut2RealTime is set up to provide behavioural feedback by sending a high or low
signal to specified pins on a GPIO breakout board connected to the computer (e.g. Adafruit rt232h). In
principle, these commands are similar to those that you might send to the pins on a Raspberry Pi.
- This code has been tested using our in-lab setup; we can make no guarantees about the code working on
your setup.
- If you wish to try a version of the code that is compatible with OpenCV webcams:
1. Use DeepLabCut to train a pose estimation model with eight points (you can train it with fewer or more points, but this
may require you to change the threshold code to avoid errors. A guide to changing the feedback threshold is coming soon!)
2. Clone this repository and run `setup.py` to install the necessary dependencies.
3. To run the code, enter your DeepLabCut environment (e.g. `activate DLC-CPU` or `activate DLC-GPU`), then:
```python
deepcut2realtime.analyze_stream(config_path, save_path, save_as_csv=True, save_frames=True, baseline=False, name=animal_name, camtype='cv2')
```
Where:
 - `config_path` is the path to your DeepLabCut model configuration file
 - `save_path` is the path to an output folder
 - `animal_name` is the name of the current animal that you are analyzing (if left blank, the animal name will be `'default_animal'`)
 - `camtype` is the type of camera you are using (`'cv2'` for standard, openCV-compatible webcam, `'sentech'` for Sentech USB3 camera
 as used in our paper).

Shortcut (using default options as listed above):
```python
deepcut2realtime.analyze_stream(config_path, save_path)
```
 
There are a number of options that you can also define:
`save_as_csv`: Choose whether or not to save the output data to CSV.

`save_frames`: Choose whether or not to save frames annotated with the bodypart position as predicted by DeepLabCut. 
This should not create significant computational overhead as it is handled in a separate thread.

`baseline`: If true, the reward signal will not be sent and the animal will not receive feedback for the user-defined behaviours.
In our study, we alternated between baseline and training trials to evaluate the animal's ability to respond only when cued to do so
(by an LED that illuminates during training trials only).

We are still working on integrating the package more robustly with DeepLabCut - please proceed with caution. To view the
original Python code used in our publication, take a look at the `original_code` folder.

If you wish to install deepcut2realtime as a native DeepLabCut method (WARNING: may cause compatibility issues!):
1. Clone this repository and, from the `manual_add_code` folder, put `predict_stream.py` and `__init.py__` in the `pose_estimation_tensorflow` subfolder
of your DeepLabCut installation. Only add `led_test.py` to this folder if you have a USB-GPIO board for delivery of feedback, otherwise
this file will return errors! All references to this Python module have been commented out in this version of `predict_stream.py`.

2.  Put `__init.py__` from the `top_level` folder, and `cli.py`, in the `deeplabcut` folder (one level up from `pose_estimation_tensorflow`).

## Dependencies
In addition to installing [DeepLabCut](https://github.com/DeepLabCut/DeepLabCut/blob/master/docs/installation.md), you need
to run:
```text
pip install pysentech pyftdi opencv-python scikit-image
```

## Credits
The code was developed by [Brandon Forys](https://github.com/bf777) and [Dongsheng Xiao](https://github.com/DongshengXiao)
with help from [Pankaj Gupta](https://github.com/pankajkgupta) at the [Murphy Lab](https://murphylab.med.ubc.ca/) 
at the University of British Columbia. It is adapted from [DeepLabCut](https://github.com/AlexEMG/DeepLabCut), which is
licensed under the [GNU Lesser General Public License v3.0](https://github.com/AlexEMG/DeepLabCut/blob/master/LICENSE).

This code was used in our _eNeuro_ article - please cite it if you plan to use it:
```text
@article {ForysENEURO.0096-20.2020,
	author = {Forys, Brandon J. and Xiao, Dongsheng and Gupta, Pankaj and Murphy, Timothy H.},
	title = {Real-Time Selective Markerless Tracking of Forepaws of Head Fixed Mice Using Deep Neural Networks},
	volume = {7},
	number = {3},
	elocation-id = {ENEURO.0096-20.2020},
	year = {2020},
	doi = {10.1523/ENEURO.0096-20.2020},
	publisher = {Society for Neuroscience},
	abstract = {Here, we describe a system capable of tracking specific mouse paw movements at high frame rates (70.17 Hz) with a high level of accuracy (mean = 0.95, SD \&lt; 0.01). Short-latency markerless tracking of specific body parts opens up the possibility of manipulating motor feedback. We present a software and hardware scheme built on DeepLabCut{\textemdash}a robust movement-tracking deep neural network framework{\textemdash}which enables real-time estimation of paw and digit movements of mice. Using this approach, we demonstrate movement-generated feedback by triggering a USB-GPIO (general-purpose input/output)-controlled LED when the movement of one paw, but not the other, selectively exceeds a preset threshold. The mean time delay between paw movement initiation and LED flash was 44.41 ms (SD = 36.39 ms), a latency sufficient for applying behaviorally triggered feedback. We adapt DeepLabCut for real-time tracking as an open-source package we term DeepCut2RealTime. The ability of the package to rapidly assess animal behavior was demonstrated by reinforcing specific movements within water-restricted, head-fixed mice. This system could inform future work on a behaviorally triggered {\textquotedblleft}closed loop{\textquotedblright} brain{\textendash}machine interface that could reinforce behaviors or deliver feedback to brain regions based on prespecified body movements.},
	URL = {https://www.eneuro.org/content/7/3/ENEURO.0096-20.2020},
	eprint = {https://www.eneuro.org/content/7/3/ENEURO.0096-20.2020.full.pdf},
	journal = {eNeuro}
}
``` 


This code is built on our past work found in [this bioRxiV preprint](https://doi.org/10.1101/482349)
and our accompanying [DeepCutRealTime repo](https://github.com/bf777/DeepCutRealTime), which you can cite here:
```text
@article {Forys482349,
	author = {Forys, Brandon and Xiao, Dongsheng and Gupta, Pankaj and Boyd, Jamie D and Murphy, Timothy H},
	title = {Real-time markerless video tracking of body parts in mice using deep neural networks},
	elocation-id = {482349},
	year = {2018},
	doi = {10.1101/482349},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {https://www.biorxiv.org/content/early/2018/11/28/482349},
	eprint = {https://www.biorxiv.org/content/early/2018/11/28/482349.full.pdf},
	journal = {bioRxiv}
}
```

`predict_stream.py` is adapted from is adapted from [`predict_videos.py`](https://github.com/DeepLabCut/DeepLabCut/blob/master/deeplabcut/pose_estimation_tensorflow/predict_videos.py)
 in DeepLabCut. The citation for DeepLabCut is:
```text
@article{Mathisetal2018,
    title={DeepLabCut: markerless pose estimation of user-defined body parts with deep learning},
    author = {Alexander Mathis and Pranav Mamidanna and Kevin M. Cury and Taiga Abe  and Venkatesh N. Murthy and Mackenzie W. Mathis and Matthias Bethge},
    journal={Nature Neuroscience},
    year={2018},
    url={https://www.nature.com/articles/s41593-018-0209-y}
}
```

`led_test.py` is a class based on [`gpio.py`](https://github.com/eblot/pyftdi/blob/master/pyftdi/tests/gpio.py) from
 the [pyftdi](https://github.com/eblot/pyftdi) library, which is covered by the [GNU Lesser General Public License v2.0](https://eblot.github.io/pyftdi/license.html).