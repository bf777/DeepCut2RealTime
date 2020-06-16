import deeplabcut
import os, sys, time

config_path = 'C:\\Users\\TM_Lab\\Desktop\\DLC2\\StreamTwoPaw2-DongshengXiao-2020-02-29\\config.yaml'
data_home_path='C:\\Users\\TM_Lab\\Desktop\\DLC2\\StreamTwoPaw\\behaviour\\alternative'
date_today=time.strftime("%Y-%m-%d", time.localtime())

os.chdir(data_home_path)
if not os.path.exists(os.path.join(data_home_path,date_today)):
    os.mkdir(os.path.join(data_home_path,date_today))


mouse_name=input('Please input the ID of this mouse:\n>>')
number_session=input('How many sessions do you want:\n>>')

for session_index in range (1, 2 * int(number_session)+1):
    print(session_index)
    if (session_index % 2) == 1:
        status='train'
        #status = 'baseline'
    else:
        status='train'
    session_direc='{}_200Hz_{}_{}'.format(mouse_name,session_index,status)
    if not  os.path.exists(os.path.join(data_home_path,date_today,session_direc)):
        os.mkdir(os.path.join(data_home_path,date_today,session_direc))
    save_path = os.path.join(data_home_path,date_today,session_direc)
    print(save_path)
    if (session_index % 2) == 1:
        deeplabcut.analyze_stream(config_path, save_path, save_as_csv=True, save_frames=True, baseline=False, name=mouse_name)
        #deeplabcut.analyze_stream(config_path, save_path, save_as_csv=True, save_frames=True, baseline=True,name=mouse_name, i_sess=session_index)
    else:
        deeplabcut.analyze_stream(config_path, save_path, save_as_csv=True, save_frames=True, baseline=False, name=mouse_name)
    print('Data saved at',session_direc)
    time.sleep(30)
