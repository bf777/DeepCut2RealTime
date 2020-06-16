"""
Adapted by:
B Forys, brandon.forys@alumni.ubc.ca
D Xiao,
P Gupta
from DeepLabCut Toolbox
https://github.com/AlexEMG/DeepLabCut
by
A Mathis, alexander.mathis@bethgelab.org
M Mathis, mackenzie@post.harvard.edu

This script analyzes a streaming video from a local webcam based
on a trained network. You need tensorflow for evaluation. Run by:
CUDA_VISIBLE_DEVICES=0 python3 predict_stream.py

PLEASE NOTE: THIS SCRIPT IS CURRENTLY ONLY COMPATIBLE WITH OMRON SENTECH USB-3 CAMERAS (we are working on adapting it
for openCV-supported webcams).
"""

####################################################
# Dependencies
####################################################
import os.path
from deeplabcut.pose_estimation_tensorflow.nnet import predict
from deeplabcut.pose_estimation_tensorflow.config import load_config
import time
import pandas as pd
import numpy as np
import os
import argparse
from pathlib import Path
import tensorflow as tf
from deeplabcut.utils import auxiliaryfunctions
import cv2
import skimage
from skimage.util import img_as_ubyte

# Real-time tracking dependencies
import _thread
from copy import deepcopy

# GPIO dependencies
from deeplabcut.utils import led_test
from pyftdi.gpio import GpioException

# Camera dependencies
from pysentech import SentechSystem


####################################################
# Loading data, and defining model folder
####################################################

def analyze_stream(config, destfolder, shuffle=1, trainingsetindex=0, gputouse=0, save_as_csv=False, save_frames=True,
                   cropping=None, baseline=True, name="default_animal"):
    """
    Makes prediction based on a trained network. The index of the trained network is specified by parameters in the config file (in particular the variable 'snapshotindex')

    You can crop the video (before analysis), by changing 'cropping'=True and setting 'x1','x2','y1','y2' in the config file. The same cropping parameters will then be used for creating the video.

    Output: The labels are stored as MultiIndex Pandas Array, which contains the name of the network, body part name, (x, y) label position \n
            in pixels, and the likelihood for each frame per body part. These arrays are stored in an efficient Hierarchical Data Format (HDF) \n
            in the same directory, where the video is stored. However, if the flag save_as_csv is set to True, the data can also be exported in \n
            comma-separated values format (.csv), which in turn can be imported in many programs, such as MATLAB, R, Prism, etc.

    Parameters
    ----------
    config : string
        Full path of the config.yaml file as a string.
    destfolder : string
        Full path of the directory to which you want to output data and (optionally) saved frames.

    shuffle: int, optional
        An integer specifying the shuffle index of the training dataset used for training the network. The default is 1.
    trainingsetindex: int, optional
        Integer specifying which TrainingsetFraction to use. By default the first (note that TrainingFraction is a list in config.yaml).

    gputouse: int, optional. Natural number indicating the number of your GPU (see number in nvidia-smi). If you do not have a GPU put None.
    See: https://nvidia.custhelp.com/app/answers/detail/a_id/3751/~/useful-nvidia-smi-queries
    save_as_csv: bool, optional
        Saves the predictions in a .csv file. The default is ``False``; if provided it must be either ``True`` or ``False``
    save_frames: bool, optional
        Labels and saves each frame of the stream to the destfolder defined above. The default is ``False``; if provided it must be either ``True`` or ``False``
    cropping: bool, optional
        Selects whether to apply cropping to each frame or not. Not recommended as it increases computational overhead.
    baseline: bool, optional
        Selects whether the current trial is a baseline trial (movement tracking but no reinforcement) or a training trial (movement tracking with reinforcement
        via water reward). If True, current trial is baseline trial; else (e.g. False), current trial is training trial.
    name: string, optional
        Pass in the name/subject ID of the animal to be observed in the current trial. This will ensure that the animal is named consistently in data output and
        trial metadata.
    Examples
    --------
    If you want to analyze a stream without saving anything
    >>> deeplabcut.analyze_stream('/analysis/project/reaching-task/config.yaml','/analysis/project/reaching-task/output')
    --------
    If you want to analyze a stream and save just the labelled frames
    >>> deeplabcut.analyze_stream('/analysis/project/reaching-task/config.yaml','/analysis/project/reaching-task/output', save_frames=True)
    --------
    If you want to analyze a stream and save both the frames and a .csv file with the coordinates of the labels
    >>> deeplabcut.analyze_stream('/analysis/project/reaching-task/config.yaml','/analysis/project/reaching-task/output', save_as_csv=True, save_frames=True)
    --------
    """
    if 'TF_CUDNN_USE_AUTOTUNE' in os.environ:
        del os.environ['TF_CUDNN_USE_AUTOTUNE']  # was potentially set during training

    if gputouse is not None:  # gpu selection
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gputouse)

    tf.reset_default_graph()
    start_path = os.getcwd()  # record cwd to return to this directory in the end

    cfg = auxiliaryfunctions.read_config(config)

    if cropping is not None:
        cfg['cropping'] = True
        cfg['x1'], cfg['x2'], cfg['y1'], cfg['y2'] = cropping
        print("Overwriting cropping parameters:", cropping)
        print("These are used for all videos, but won't be save to the cfg file.")

    trainFraction = cfg['TrainingFraction'][trainingsetindex]

    modelfolder = os.path.join(cfg["project_path"], str(auxiliaryfunctions.GetModelFolder(trainFraction, shuffle, cfg)))
    path_test_config = Path(modelfolder) / 'test' / 'pose_cfg.yaml'
    try:
        dlc_cfg = load_config(str(path_test_config))
    except FileNotFoundError:
        raise FileNotFoundError(
            "It seems the model for shuffle %s and trainFraction %s does not exist." % (shuffle, trainFraction))

    # Check which snapshots are available and sort them by # iterations
    try:
        Snapshots = np.array(
            [fn.split('.')[0] for fn in os.listdir(os.path.join(modelfolder, 'train')) if "index" in fn])
    except FileNotFoundError:
        raise FileNotFoundError(
            "Snapshots not found! It seems the dataset for shuffle %s has not been trained/does not exist.\n Please train it before using it to analyze videos.\n Use the function 'train_network' to train the network for shuffle %s." % (
            shuffle, shuffle))

    if cfg['snapshotindex'] == 'all':
        print(
            "Snapshotindex is set to 'all' in the config.yaml file. Running video analysis with all snapshots is very costly! Use the function 'evaluate_network' to choose the best the snapshot. For now, changing snapshot index to -1!")
        snapshotindex = -1
    else:
        snapshotindex = cfg['snapshotindex']

    increasing_indices = np.argsort([int(m.split('-')[1]) for m in Snapshots])
    Snapshots = Snapshots[increasing_indices]

    print("Using %s" % Snapshots[snapshotindex], "for model", modelfolder)

    ##################################################
    # Load and setup CNN part detector
    ##################################################

    # Check if data already was generated:
    dlc_cfg['init_weights'] = os.path.join(modelfolder, 'train', Snapshots[snapshotindex])
    trainingsiterations = (dlc_cfg['init_weights'].split(os.sep)[-1]).split('-')[-1]

    # update batchsize (based on parameters in config.yaml)
    dlc_cfg['batch_size'] = cfg['batch_size']
    # Name for scorer:
    DLCscorer = auxiliaryfunctions.GetScorerName(cfg, shuffle, trainFraction, trainingsiterations=trainingsiterations)

    sess, inputs, outputs = predict.setup_pose_prediction(dlc_cfg)
    pdindex = pd.MultiIndex.from_product([[DLCscorer], dlc_cfg['all_joints_names'], ['x', 'y', 'likelihood']],
                                         names=['scorer', 'bodyparts', 'coords'])

    if gputouse is not None:  # gpu selectinon
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gputouse)

    ##################################################
    # Set up data buffer and global variables to be used in threads
    ##################################################
    global PredicteData
    PredicteData = np.zeros((50000, 3 * len(dlc_cfg['all_joints_names'])))
    global led_arr
    led_arr = np.zeros((50000, 7))
    global x_range
    global y_range
    global acc_range
    x_range = list(range(0, (3 * len(dlc_cfg['all_joints_names'])), 3))
    y_range = list(range(1, (3 * len(dlc_cfg['all_joints_names'])), 3))
    acc_range = list(range(2, (3 * len(dlc_cfg['all_joints_names'])), 3))
    global colors
    colors = [(0, 0, 255), (0, 165, 255), (0, 255, 255), (0, 255, 0), (255, 0, 0), (240, 32, 160), (0, 0, 255),
              (0, 165, 255)]
    global empty_count
    empty_count = 0
    global threshold_count
    AnalyzeStream(DLCscorer, trainFraction, cfg, dlc_cfg, sess, inputs, outputs, pdindex, save_as_csv, save_frames,
                  destfolder, name, baseline)

##################################################
# We define our own two-frame batches for real-time tracking, so we don't use builtin batchwise pose prediction.
##################################################
# def GetPoseF(cfg, dlc_cfg, sess, inputs, outputs, cap, nframes, batchsize):
#     ''' Batchwise prediction of pose '''
#
#     PredicteData = np.zeros((50000, 3 * len(dlc_cfg['all_joints_names'])))  # Preallocates memory to be filled by stream
#     batch_ind = 0  # keeps track of which image within a batch should be written to
#     batch_num = 0  # keeps track of which batch you are at
#     ny, nx = int(cap.get(4)), int(cap.get(3))
#     if cfg['cropping']:
#         print(
#             "Cropping based on the x1 = %s x2 = %s y1 = %s y2 = %s. You can adjust the cropping coordinates in the config.yaml file." % (
#             cfg['x1'], cfg['x2'], cfg['y1'], cfg['y2']))
#         nx = cfg['x2'] - cfg['x1']
#         ny = cfg['y2'] - cfg['y1']
#         if nx > 0 and ny > 0:
#             pass
#         else:
#             raise Exception('Please check the order of cropping parameter!')
#         if cfg['x1'] >= 0 and cfg['x2'] < int(cap.get(3) + 1) and cfg['y1'] >= 0 and cfg['y2'] < int(cap.get(4) + 1):
#             pass  # good cropping box
#         else:
#             raise Exception('Please check the boundary of cropping!')
#
#     frames = np.empty((batchsize, ny, nx, 3), dtype='ubyte')  # this keeps all frames in a batch
#     while (cap.isOpened()):
#         ret, frame = cap.read()
#         if ret:
#             frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             if cfg['cropping']:
#                 frames[batch_ind] = img_as_ubyte(frame[cfg['y1']:cfg['y2'], cfg['x1']:cfg['x2']])
#             else:
#                 frames[batch_ind] = img_as_ubyte(frame)
#
#             if batch_ind == batchsize - 1:
#                 pose = predict.getposeNP(frames, dlc_cfg, sess, inputs, outputs)
#                 PredicteData[batch_num * batchsize:(batch_num + 1) * batchsize, :] = pose
#                 batch_ind = 0
#                 batch_num += 1
#             else:
#                 batch_ind += 1
#         else:
#             nframes = counter
#             print("Detected frames: ", nframes)
#             if batch_ind > 0:
#                 pose = predict.getposeNP(frames, dlc_cfg, sess, inputs,
#                                          outputs)  # process the whole batch (some frames might be from previous batch!)
#                 PredicteData[batch_num * batchsize:batch_num * batchsize + batch_ind, :] = pose[:batch_ind, :]
#             break
#         counter += 1
#     return PredicteData


def GetPoseS(cfg, dlc_cfg, sess, inputs, outputs, cap, w, h, nframes, save_frames, destfolder, baseline):
    """ Non batch wise pose estimation for video cap."""
    # Prepare data arrays
    global lastFrameWasMoved
    global thisFrameWasMoved
    global threshold_count
    lastFrameWasMoved = False
    thisFrameWasMoved = False
    x_arr = []
    y_arr = []
    x_overall = []
    y_overall = []
    x_overall_left = []
    x_overall_right = []
    y_overall_left = []
    y_overall_right = []
    threshold = 0
    LED_arr = []
    time_arr = []
    time_index = 0
    if cfg['cropping']:
        print(
            "Cropping based on the x1 = %s x2 = %s y1 = %s y2 = %s. You can adjust the cropping coordinates in the config.yaml file." % (
            cfg['x1'], cfg['x2'], cfg['y1'], cfg['y2']))
        if w > 0 and h > 0:
            pass
        else:
            raise Exception('Please check the order of cropping parameter!')

    # Set up GPIO board
    LED = led_test.LEDTest()
    mask = 0xFF
    LED.open(mask)
    # Tries to initialize all GPIO ports used in code to low state (0) so that they won't accidentally turn on at the start
    # of the trial.
    try:
        LED.set_gpio(5, False)
        LED.set_gpio(6, False)
        LED.set_gpio(7, False)
    except GpioException:
        pass
    start = time.time()
    counter = 0
    threshold_count = 0
    frame_arr = []
    try:
        while cap:
            frame = cap.grab_frame()
            # ret, frame = cap.read()
            frame = frame.as_numpy()
            frame = np.uint8(frame)
            # if ret:
            if frame.any():
                frame = skimage.color.gray2rgb(frame)
                if cfg['cropping']:
                    frame = img_as_ubyte(frame[cfg['y1']:cfg['y2'], cfg['x1']:cfg['x2']])
                else:
                    frame = img_as_ubyte(frame)
                frame_arr.append(frame)
                frame_time = time.time()
                led_arr[counter, 0] = frame_time
                led_arr[counter + 1, 0] = frame_time
                if (time.time() - start) > 0:
                    print("Current FPS: {} fps, Time elapsed: {} s, Number of frames so far: {}".format(
                        round(counter / (time.time() - start), 2),
                        round(time.time() - start, 2), counter), end='\r')
                if len(frame_arr) == 2:
                    # This thread carries out pose estimation on each batch of two frames that arrives from the camera.
                    _thread.start_new_thread(frame_process, (frame_arr, dlc_cfg, sess, inputs, outputs, counter,
                                                             save_frames, destfolder, LED, x_arr, y_arr, frame_time,
                                                             start, baseline))
                    frame_arr = []
                    counter += 2
            x_arr = []
            y_arr = []
            x_first = 0
            y_first = 0
            threshold = 0
            # Run each trial for 130 seconds
            if time.time() - start >= 130:
                nframes = counter
                break
    except KeyboardInterrupt:
        _thread.exit()
        print("Finished.")
        cap.release()
        exit()
    LED.close()
    return nframes


def AnalyzeStream(DLCscorer, trainFraction, cfg, dlc_cfg, sess, inputs, outputs, pdindex, save_as_csv, save_frames,
                  destfolder, name, baseline):
    """Sets up camera connection for pose estimation, and handles data output."""
    # Setup camera connection
    # REPLACE WITH PATH TO YOUR SENTECH CAMERA SDK!
    sdk_location = r"C:\Users\TM_Lab\Desktop\Greg_desktop\StCamUSBPack_EN_190207\3_SDK\StandardSDK(v3.14)"
    system = SentechSystem(sdk_location)
    cam = system.get_camera(0)
    print("Camera connected! The camera model is " + str(cam.model.decode("utf-8")))

    print("Starting to analyze stream")
    # Accept a single connection and make a file-like object out of it
    cap = cam
    dataname = os.path.join(destfolder, DLCscorer + '_' + name + '.h5')
    dataname_led = os.path.join(destfolder, DLCscorer + '_' + name + '_LED.h5')
    led_data_cols = ['FrameTime', 'MovementDiffLeft', 'MovementDiffRight', 'ThresholdTime', 'Delay', 'FlashTime',
                     'WaterTime']
    size = (int(cap.image_shape[0]), int(cap.image_shape[1]))
    # size = (int(cap.get(3)), int(cap.get(4)))
    w, h = size
    shutter = 1/500
    brightness = 68
    v_blanking = 982
    acc_tolerance = 0.20
    missing_count = 0
    nframes = 0

    print("Starting to extract posture")
    start = time.time()
    nframes = GetPoseS(cfg, dlc_cfg, sess, inputs, outputs, cap, w, h, nframes, save_frames, destfolder, baseline)

    # stop the timer and display FPS information
    stop = time.time()
    fps = nframes / (stop - start)
    print("\n")
    print("[INFO] elasped time: {:.2f}".format(stop - start))
    print("[INFO] approx. FPS: {:.2f}".format(fps))

    # If there's rows with blank data at the end of the trial, record this data as missing/dropped frames
    for row in PredicteData[int(np.around(fps * 10)):nframes, :]:
        if 0 in row:
            missing_count += 1

    time.sleep(10)
    avg_array = led_arr[:, 4]
    avg_delay = avg_array[avg_array != 0].mean()
    sd_delay = np.std(avg_array[avg_array != 0])
    avg_acc = PredicteData[:nframes, acc_range].mean()

    # Prints out results of trial
    print("Empty values: {}, {} per second".format(str(missing_count), str(missing_count / (stop - start))))
    print("Adjusted frame rate: {}".format(str((nframes - missing_count) / (stop - start))))
    print("Average delay: {} s".format(str(avg_delay)))
    print("Standard dev. of delay: {} s".format((str(sd_delay))))
    print("Average tracking accuracy: {}".format((str(avg_acc))))

    # Save metadata with trial information and camera information
    dictionary = {
        "name": name,
        "start": start,
        "stop": stop,
        "run_duration": stop - start,
        "Scorer": DLCscorer,
        "DLC-model-config file": dlc_cfg,
        "fps": fps,
        "fps_adjusted": ((nframes - missing_count) / (stop - start)),
        "avg_delay": avg_delay,
        "sd_delay": sd_delay,
        "v_blanking": v_blanking,
        "shutter": shutter,
        "brightness": brightness,
        "batch_size": dlc_cfg["batch_size"],
        "frame_dimensions": (h, w),
        "nframes": nframes,
        "acc_tolerance": acc_tolerance,
        "avg_acc": avg_acc,
        "iteration (active-learning)": cfg["iteration"],
        "training set fraction": trainFraction,
        "cropping": cfg['cropping'],
        "LED_time": 0.2,
        "water_time": 0.15,
        "refractory_period": 0.3
    }
    metadata = {'data': dictionary}

    print("Saving results in {} and {}".format(dataname, dataname_led))
    auxiliaryfunctions.SaveData(PredicteData[:nframes, :], metadata, dataname, pdindex, range(nframes), save_as_csv)
    auxiliaryfunctions.SaveData(led_arr[:nframes, :], metadata, dataname_led, led_data_cols, range(nframes),
                                save_as_csv)
    cam.release()


####################################################
# GPIO Helper functions
####################################################
def frame_process(frame_arr, dlc_cfg, sess, inputs, outputs, counter, save_frames, destfolder, LED, x_arr, y_arr,
                  frame_time, start, baseline):
    """Estimates pose in each frame, and optionally plots the pose on each frame."""
    # Set up parameters for refractory period (to check whether there was movement on the previous frame).
    global thisFrameWasMoved
    global lastFrameWasMoved
    global threshold_time
    global threshold_count
    x_avg_left = []
    y_avg_left = []
    x_avg_right = []
    y_avg_right = []
    # Run DeepLabCut pose prediction on each batch of two frames
    for n, frame in enumerate(frame_arr):
        pose = predict.getpose(frame, dlc_cfg, sess, inputs, outputs)
        if time.time() - start >= 10:
            PredicteData[counter + n, :] = pose.flatten()  # NOTE: thereby cfg['all_joints_names'] should be same order as bodyparts!
            # Create lists to store average x and y paw positions
            for x_val, y_val in zip(x_range, y_range):
                x_arr.append(PredicteData[counter + n, :][x_val])
                y_arr.append(PredicteData[counter + n, :][y_val])
            if n == 0:
                add = 0
            elif n == 1:
                add = 8
            x_avg_right.append(np.mean(x_arr[0 + add:3 + add]))
            x_avg_left.append(np.mean(x_arr[4 + add:7 + add]))
            y_avg_right.append(np.mean(y_arr[0 + add:3 + add]))
            y_avg_left.append(np.mean(y_arr[4 + add:7 + add]))
            # Save the frames in a separate thread (if we wish to do so)
            if save_frames:
                _thread.start_new_thread(frame_save_func, (frame, x_range, y_range, x_avg_left, y_avg_left,
                                                           x_avg_right, y_avg_right, destfolder, n, counter))
    # THRESHOLDS FOR Y MOVEMENT
    # min y movement on left paw to be counted as a trigger = 5 px
    # max y movement on right paw to be counted as a trigger = 10 px
    # Max y movement overall: 100 px
    y_left = 5
    y_right = 10
    y_upper_lim = 100

    # Red LED flash timing (s)
    gpio_light_time = 0.2

    # Length of trial (for data recording purposes)
    trial_length = 130

    # Water reward release time (s)
    gpio_water_time = 0.15

    # Refractory time (s): time after last trigger during which no LED flash can be triggered/reward given
    refractory_time = 0.3

    thisFrameWasMoved = False

    # We wait for 10.1 s after the start of the trial to start giving behavioural feedback. This is because the latency
    # for pose estimation is longer, and the framerate is less stable, in the first 10 s of the trial
    # (likely because of Python library setup).
    if counter >= 1 and time.time() - start > 10.1:
        trial_start = True
        if not baseline:
            _thread.start_new_thread(led_task, (LED, trial_start, trial_length - 10)),
        led_arr[counter, 1] = abs(y_avg_left[1] - y_avg_left[0])
        led_arr[counter, 2] = abs(y_avg_right[1] - y_avg_right[0])
        if (y_left <= abs(y_avg_left[1] - y_avg_left[0]) <= y_upper_lim and
                y_right >= abs(y_avg_right[1] - y_avg_right[0])):
            # and PredicteData[counter, acc_range].mean() >= 0.20
            thisFrameWasMoved = True
            if threshold_count == 0:
                threshold_time = time.time() - 0.5
        if thisFrameWasMoved and not lastFrameWasMoved and abs(time.time() - threshold_time) >= refractory_time:
            threshold_time = time.time()
            threshold_count += 1
            delay = threshold_time - frame_time
            led_arr[counter + 1, 3] = threshold_time
            led_arr[counter + 1, 4] = delay
            ttt = True
            _thread.start_new_thread(thresholdFunc, (LED, baseline, ttt, gpio_light_time, gpio_water_time, counter,
                                                     threshold_time))
        lastFrameWasMoved = thisFrameWasMoved

    # Blank out lists and other variables that are changed for each trigger
    x_arr = []
    y_arr = []
    x_avg_left = []
    y_avg_left = []
    x_avg_right = []
    y_avg_right = []
    x_first = 0
    y_first = 0
    threshold = 0


def led_task(LED, reached, trial_length):
    """Starts GPIO (green LED) output to indicate to the mouse when the trial has started."""
    if reached:
        # print("Reached")
        # Initiates LED
        try:
            LED.set_gpio(6, True)
        except GpioException:
            pass
            # print("\nError switching light on!")
        time.sleep(trial_length)
        try:
            LED.set_gpio(6, False)
        except GpioException:
            pass
            # print("\nError switching light off!")


def thresholdFunc(LED, baseline, reached, gpio_light_time, gpio_water_time, counter, threshold_time):
    '''Controls GPIO (red LED and (water pump) output based on calculated threshold.'''
    if reached:
        flash_time = time.time()
        led_arr[counter + 1, 5] = flash_time
        water_time = time.time()
        led_arr[counter + 1, 6] = water_time
        # Initiates LED
        try:
            LED.set_gpio(5, True)  # LED start
            if not baseline:
                LED.set_gpio(7, True)  # water start
        except GpioException:
            pass
        time.sleep(gpio_water_time)
        try:
            if not baseline:
                LED.set_gpio(7, False)  # water end
        except GpioException:
            pass
        time.sleep(abs(gpio_light_time - gpio_water_time))
        try:
            LED.set_gpio(5, False)  # LED end
        except GpioException:
            pass


def frame_save_func(frame, x_range, y_range, x_avg_left, y_avg_left, x_avg_right, y_avg_right, destfolder, n, counter):
    '''Handles plotting estimated paw positions on frames, and saving these frames.'''
    tmp = deepcopy(frame)
    avgs = False
    if avgs:
        cv2.circle(tmp, (int(x_avg_right[n]), int(y_avg_right[n])), 6, (0, 165, 255), -1)
        cv2.circle(tmp, (int(x_avg_left[n]), int(y_avg_left[n])), 6, (0, 165, 255), -1)
    else:
        for x_plt, y_plt, c in zip(x_range, y_range, colors):
            cv2.circle(tmp, (int(PredicteData[counter + n, :][x_plt]),
                             int(PredicteData[counter + n, :][y_plt])), 2, c, -1)
    cv2.imwrite(os.path.join(destfolder, 'frame{}.png'.format(str(counter + n))), tmp)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    cli_args = parser.parse_args()
