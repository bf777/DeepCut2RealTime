# DeepCut2RealTime
Welcome to DeepCut2RealTime, an add-on for DeepLabCut that enables real-time tracking
and reinforcement of animal behaviours. This code was used to carry out the behavioural experiments outlined
in Forys, Xiao, Gupta, and Murphy (under review). It builds upon the code outlined in [Forys, Xiao, Gupta,
Boyd, and Murphy (2018)](https://doi.org/10.1101/482349).

## Features:
- Integrates with DeepLabCut 2.0.6 and later through an easy-to-use iPython interface.
- Combines real-time application of DeepLabCut's pose estimation framework at high
framerates (> 90 Hz) with customizable selection of behaviours to reinforce.

## Important notes:
- At this time, DeepCut2RealTime has only been tested with the Sentech STC-MCCM401U3V USB-3 Vision camera; we are
working on making it compatible with standard OpenCV cameras as well.
- DeepCut2RealTime is set up to provide behavioural feedback by sending a high or low
signal to specified pins on a GPIO breakout board connected to the computer (e.g. Adafruit rt232h). In
principle, these commands are similar to those that you might send to the pins on a Raspberry Pi.
- This code has been tested using our in-lab setup; we can make no guarantees about the code working on
your setup.

## Dependencies
In addition to installing [DeepLabCut](https://github.com/AlexEMG/DeepLabCut/blob/master/docs/installation.md), you need
to run:
```text
pip install pysentech pyftdi opencv-python scikit-image
```

## Credits
The code was developed by [Brandon Forys](https://github.com/bf777) and [Dongsheng Xiao](https://github.com/DongshengXiao)
with help from [Pankaj Gupta](https://github.com/pankajkgupta) at the [Murphy Lab](https://murphylab.med.ubc.ca/) 
at the University of British Columbia. It is adapted from [DeepLabCut](https://github.com/AlexEMG/DeepLabCut), which is
licensed under the [GNU Lesser General Public License v3.0](https://github.com/AlexEMG/DeepLabCut/blob/master/LICENSE).

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

`predict_stream.py` is adapted from is adapted from [`predict_videos.py`](https://github.com/AlexEMG/DeepLabCut/blob/master/deeplabcut/pose_estimation_tensorflow/predict_videos.py)
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