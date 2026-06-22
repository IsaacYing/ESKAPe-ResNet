import serial,os,glob,datetime,pickle as pk
import sys
import time
import numpy as np,pandas as pd
import cv2
from ipywidgets import *
from IPython.display import display,clear_output
import tifffile as tif
from ipyfilechooser import FileChooser
from skimage import transform
import threading
import time,timeit
from matplotlib import pyplot as plt
from matplotlib.path import Path
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, GlobalAveragePooling2D, Rescaling, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
import keras
import numpy as np
from scipy.interpolate import griddata
from scipy.interpolate import Rbf

global stop_loop
stop_loop = False 

class MG120_handler:
    
    def __init__(self,PORT = 'COM8'):
        self.port = PORT
        self.BAUD_RATE = 9600
        self.packet_header = b'\x5A\xA5'
        self.packet_length = b'\x04'  # 假设包长度为4字节
        self.write_code = b'\x80'
        self.onoff_addresses = {'R':b'\x10','G':b'\x12','B':b'\x14','UV':b'\x16'}
        self.power_addresses = {'R':b'\x11','G':b'\x13','B':b'\x15','UV':b'\x17'}
        self.data_length = b'\x01'  # 假设起始地址为1字节
        self.ser = serial.Serial(self.port, self.BAUD_RATE, 
                                    bytesize=serial.EIGHTBITS,stopbits=serial.STOPBITS_ONE,
                                    timeout=1)
        self.channels = ['R','G','B','UV']
        self.is_vacant=True
        for k,v in self.onoff_addresses.items():
            self.channel_off(k)
        
    def channel_on(self,channel='R'):
        self.is_vacant=False
        self.ser.write(self.packet_header +\
                   self.packet_length +\
                   self.write_code +\
                   self.onoff_addresses[channel] + self.data_length + b'\x01')

    def channel_off(self,channel='R'):
        self.ser.write(self.packet_header +\
                   self.packet_length +\
                   self.write_code +\
                   self.onoff_addresses[channel] + self.data_length + b'\x00')
        self.is_vacant=True

    def channel_power(self,channel='R',power=50):
        self.ser.write(self.packet_header +\
                   self.packet_length +\
                   self.write_code +\
                   self.power_addresses[channel] + self.data_length + int_to_hex_bytearray(power))

    def pulse(self,channel='R', #channel code
              power=50,
              duration=100, # duration in miliseconds
              ):
        if channel not in self.onoff_addresses:
            print('Incorrect channel name, only {} are supported'.format(list(self.onoff_addresses.keys())))
        else:
            for k,v in self.onoff_addresses.items():
                if k!=channel:
                    self.channel_off(k)
        self.channel_power(channel,power)
        self.channel_on(channel)
        time.sleep(duration/1000)
        self.channel_off(channel)

    def turnoff_all(self):
        for c in self.channels:
            self.channel_off(c)
            
    def end_session(self,verbose=False):
        if self.is_vacant:
            self.turnoff_all()
            self.ser.close()

def int_to_hex_bytearray(num):
    if 0 <= num <= 100:
        # 将整数转换为十六进制字符串，并格式化为'0x'形式
        hex_str = f'0x{num:02x}'
        # 将格式化后的字符串转换为字节数组
        return bytes.fromhex(hex_str[2:])  # 去掉'0x'前缀
    else:
        raise ValueError("输入的整数必须在0到100之间")

class NikonTiControlPanel:

    def __init__(self,mm_core,mm_studio,mg_handler):
        self.core = mm_core
        self.mg = mg_handler
        self.studio = mm_studio
        self.default_filesave_path = "D:/Zhulab_microscopy_data/autosave"
        self.default_afmodel_path = "D:/Zhulab_microscopy_data/autofocus_models"
        self.microscope_config_panel = TiConfig_panel()
        if not os.path.isdir(self.default_filesave_path):
            os.mkdir(self.default_filesave_path)
        if not os.path.isdir(self.default_afmodel_path):
            os.mkdir(self.default_afmodel_path)
        self.filesave_pathselect = FileChooser(self.default_filesave_path,default_path=self.default_filesave_path,select_default=True,
                                               title='<b>Auto-save path<b>') 
        self.afmodel_pathselect = FileChooser(self.default_afmodel_path,default_path=self.default_afmodel_path,select_default=True,
                                              title='<b>Select auto-focus model<b>') 
        self.mg_gui = MG_GUI(self.mg,self.core)
        self.zstack_panel = ZStackPanel()
        self.tilescan_panel = TileScanPanel()
        self.channel_panel = ChannelSelectPanel()
        self.live_view_panel = LiveViewButton(self.studio,self.core)
        self.live_bf_panel = LiveViewButton(self.studio,self.core,'Live BF',auto_bf=True)
        self.plate_calib_panel = PlateScanPanel(self.core)
        self.exposure_panel = AdjustExposure(self.core)
        self.model_af_panel = Model_AF_Panel()
        self.scan_af_panel = ZScan_AF_Panel()
        self.md_acquisition_panel = MD_Acquisition_Panel(self)
        self.layout = VBox([HBox([VBox([HBox([self.mg_gui.layout,
                                              VBox([self.live_view_panel.live_button,self.live_bf_panel.live_button,self.exposure_panel.layout]),
                                              VBox([self.filesave_pathselect,self.afmodel_pathselect])]),
                                        HBox([self.channel_panel.layout,self.microscope_config_panel.layout]),
                                        HBox([self.zstack_panel.layout,self.tilescan_panel.layout,self.md_acquisition_panel.layout])],
                                        layout = Layout(width='80%',border='1px solid')),
                                  VBox([self.model_af_panel.layout,self.scan_af_panel.layout],layout = Layout(width='20%',border='1px solid'))]),
                            self.plate_calib_panel.layout])


class TiConfig_panel:

    def __init__(self,
                  camera_props = {'Dhyana 401D':{'pixel_size':6.5,'width':2048,'height':2048}},
                  objective_props = {'60x Oil':{'magnification':60,'NA':1.42},
                                     '100x Oil':{'magnification':100,'NA':1.45},
                                     '40x Air':{'magnification':40,'NA':0.95},
                                     '20x Air':{'magnification':20,'NA':0.75}},
                  condensor_turret = ['N/A','Ph-1','Ph-3']):
        self.camera_props=camera_props
        self.objective_props = objective_props
        self.condensor_turret = condensor_turret
        self.camera_dropdown = Dropdown(description='Camera',
                                        options=list(camera_props.keys()),
                                        value=list(camera_props.keys())[0])
        self.objective_dropdown = Dropdown(description='Obj lens',
                                        options=list(objective_props.keys()),
                                        value=list(objective_props.keys())[0])
        self.condensor_dropdown = Dropdown(description='Condensor',
                                           options=condensor_turret,
                                           value=condensor_turret[0])
        self.get_current_state()
        
        self.camera_dropdown.observe(self.update_dropdown, names='value')
        self.objective_dropdown.observe(self.update_dropdown, names='value')
        self.condensor_dropdown.observe(self.update_dropdown, names='value')
        self.layout = VBox([self.camera_dropdown,
                            self.objective_dropdown,
                            self.condensor_dropdown],layout=Layout(width='30%',border='1px solid'))
        
    def update_dropdown(self,change):
        self.get_current_state()
            
    def get_current_state(self):
        self.camera = self.camera_dropdown.value
        self.camera_pixel_size = self.camera_props[self.camera]['pixel_size']
        self.camera_width = self.camera_props[self.camera]['width']
        self.camera_height = self.camera_props[self.camera]['height']
        self.objective_lens = self.objective_dropdown.value
        self.magnification = self.objective_props[self.objective_lens]['magnification']
        self.na = self.objective_props[self.objective_lens]['NA']
        self.pixel_microns = round(self.camera_pixel_size/self.magnification,6)
        self.image_size = [self.camera_height*self.pixel_microns, self.camera_width*self.pixel_microns]
        self.condensor = self.condensor_dropdown.value


    

class MD_Acquisition_Panel:

    def __init__(self,
                 TI_controller):
        self.controller = TI_controller
        self.camera_pixel_size=self.controller.microscope_config_panel.camera_pixel_size
        self.magnification = self.controller.microscope_config_panel.magnification
        self.pixel_microns = self.controller.microscope_config_panel.pixel_microns
        self.counter=0
        self.axes = 'TZCYX'
        self.core = self.controller.core
        self.af_model = None
        self.deep_AF = False
        self.led = self.controller.mg
        self.width = self.core.get_image_width()
        self.height = self.core.get_image_height()
        self.camera = self.core.get_camera_device()
        self.xy_device = self.core.get_xy_stage_device()
        self.z_device = self.core.get_focus_device()
        self.shutter = self.core.get_shutter_device()
        self.autosave_label = Label('Autosave fileheader:',style = {'font_style':'bold'})
        self.autosave_textpanel = Text(value='{}_autosave'.format(time_stamp().split(' ')[0].replace('-','')),
                                       placeholder='Type something',
                                       description='',disabled=False)
        self.snap_button = Button(description='One-shot snap',disabled=False,style = {'button_color':'lightgreen'},
                                  tooltip='Click me',layout = Layout(width='15pix',border='1px dashed'))
        self.md_acquisition_button = Button(description='Run experiment',disabled=False,style = {'button_color':'lightgreen'},
                                     tooltip='Click me',layout = Layout(width='15pix',border='1px dashed'))
        self.stop_acquisition_button = Button(description='End experiment',style = {'button_color':'salmon'},disabled = True, 
                                              tooltip='Click me',layout = Layout(width='15pix',border='1px dashed'))
        self.live_after_acq = Checkbox(value=True,description='Post-run live',disabled=False,indent=False,layout=Layout(width='90%'))
        self.position_af_method = Dropdown(description='',
                                            options=['None','SShot','Scan','SShot then scan'],
                                            value='None',layout=Layout(width='90%'))
        self.snap_button.on_click(self.single_shot_snap())
        self.md_acquisition_button.on_click(self.on_start_button_clicked)
        self.stop_acquisition_button.on_click(self.on_stop_button_clicked)
        
        self.layout = VBox([self.autosave_label,
                            self.autosave_textpanel,
                            HBox([VBox([self.snap_button,self.md_acquisition_button,self.stop_acquisition_button],layout=Layout(width='50%')),
                                  VBox([Label('AF method:',layout=Layout(width='90%')),self.position_af_method,self.live_after_acq],layout=Layout(width='50%'))])],
                            layout= Layout(width='30%',border='1px solid'))
        
    def on_start_button_clicked(self, b):
        global stop_loop
        stop_loop = False  # re-initialize 
        threading.Thread(target=self.run_md_acquisition).start()

    def on_stop_button_clicked(self, b):
        global stop_loop
        stop_loop = True
        
    def run_md_acquisition(self):
        global stop_loop
        #
        if self.controller.afmodel_pathselect.selected.endswith('.pk'):
            self.af_model = pk.load(open(self.controller.afmodel_pathselect.selected,'rb'))
        elif self.controller.afmodel_pathselect.selected.endswith('.keras'):
            self.af_model = keras.saving.load_model(self.controller.afmodel_pathselect.selected)
            self.deep_AF = True
        else:
            self.af_model = pk.load(open("D:/Zhulab_microscopy_data/autofocus_models/20240703_AFmodel1.pk",'rb'))

        if self.stop_acquisition_button.disabled == True:
            self.stop_acquisition_button.disabled = False
            
        self.plate_scan_positions = [None]
        self.z_scan_positions = [None]
        self.tile_scan_offset = [None]
        axes = 'CYX' 
        timestamps = ''
        self.controller.live_view_panel.umanager_live_snap.set_live_mode_on(False)
        channels = self.controller.channel_panel.get_channel_info()
        
        if self.controller.plate_calib_panel.do_platescan.value:
            nrows = self.controller.plate_calib_panel.n_rows
            ncols = self.controller.plate_calib_panel.n_cols
            for r in range(nrows): #change to ...range(4)
                for _c in range(ncols):
                    if r%2==1:
                        c = ncols-1-_c   
                    else:
                        c = _c
                    well_name = '{}{}'.format(chr(65+r),str(c+1).zfill(2))
                    x0,y0 = self.controller.plate_calib_panel.plate.get_well_coords(well_name)
                    self.plate_scan_positions.append([well_name,x0,y0])
        if self.controller.zstack_panel.do_z_scan.value:
            axes = 'Z' + axes
            up = self.controller.zstack_panel.up_scan.value
            down = self.controller.zstack_panel.down_scan.value
            nsteps = self.controller.zstack_panel.scan_steps.value
            for z in np.linspace(down,up,nsteps):
                self.z_scan_positions.append(z)

        if self.controller.tilescan_panel.do_tilescan.value:
            tile_rows = self.controller.tilescan_panel.n_rows.value
            tile_cols = self.controller.tilescan_panel.n_cols.value
            percentage_overlap = self.controller.tilescan_panel.overlap.value
            pos_dist = self.width*self.pixel_microns*(1-percentage_overlap*0.01)
            self.tile_scan_offset += [t for t in generate_xy_rel(tile_rows,tile_cols,pos_dist)]

        if len(self.plate_scan_positions) > 1:
            self.plate_scan_positions = self.plate_scan_positions[1:]
        if len(self.z_scan_positions) > 1:
            self.z_scan_positions = self.z_scan_positions[1:]
        if len(self.tile_scan_offset) > 1:
            self.tile_scan_offset = self.tile_scan_offset[1:]
        
        # run experiment
        skip=False
        for _w in self.plate_scan_positions:
            if stop_loop:
                print("Experiment was manually terminated at {}".format(time_stamp()))
                break
            if _w is not None:
                well,x0,y0 = _w
                well_name = 'Well-{}_'.format(well)
                if self.controller.plate_calib_panel.plate_control_panel.checkbox_dict[well].value:
                    skip=False
                    self.core.set_xy_position(x0,y0)
                    self.core.wait_for_device(self.xy_device)
                    if self.controller.plate_calib_panel.magellan_model is not None and self.controller.plate_calib_panel.use_magellan.value:
                        z0 = float(self.controller.plate_calib_panel.predict_z_magellan(x0,y0))
                        self.core.set_position(z0)
                        self.core.wait_for_device(self.z_device)
                    if 'can' in self.controller.md_acquisition_panel.position_af_method.value:
                        scand, z_list = AF_z_max_v(self.core,
                                                   self.led,
                                                   num_z_slices = 9,
                                                   light_src=self.controller.scan_af_panel.channel_dropdown.value,
                                                   power=self.controller.scan_af_panel.power.value,
                                                   exposure=self.controller.scan_af_panel.set_exposure.value,
                                                   z_offset = [-9,9])
                        best_z = z_list[np.argmax([img2freq(cv2_rescale(x,0.25)).mean() for x in scand])]
                        self.core.set_position(best_z)
                        self.core.wait_for_device(self.z_device)
                    if 'SShot' in self.controller.md_acquisition_panel.position_af_method.value:
                        iterative_auto_focus(self.controller,
                                             self.af_model,
                                             self.deep_AF,
                                             zrange = [self.controller.model_af_panel.zmin.value,
                                                       self.controller.model_af_panel.zmax.value],
                                             z_steps=self.controller.model_af_panel.zsteps.value,
                                             exposure=self.controller.model_af_panel.set_exposure.value)
                else:
                    skip=True
            else:
                well_name,x0,y0 = '',self.core.get_x_position(),self.core.get_y_position()
                skip=False
            if not skip:
                group_z = []
                for pos,_t in enumerate(self.tile_scan_offset):
                    if stop_loop:
                        print("Experiment was manually terminated at {}".format(time_stamp()))
                        break
                    if _t is not None:
                        dx,dy = _t
                        x,y = x0+dx,y0+dy
                        self.core.set_xy_position(x,y)
                        self.core.wait_for_device(self.xy_device)
                    else:
                        dx,dy = 0,0
                    if 'SShot' in self.controller.md_acquisition_panel.position_af_method.value:
                        iterative_auto_focus(self.controller,
                                             self.af_model,
                                             self.deep_AF,
                                             zrange = [-2,0],
                                             z_steps=2,
                                             exposure=self.controller.model_af_panel.set_exposure.value)
                    if 'can' in self.controller.md_acquisition_panel.position_af_method.value:
                        scand, z_list = AF_z_max_v(self.core,
                                                   self.led,
                                                   num_z_slices = self.controller.scan_af_panel.zsteps.value,
                                                   light_src=self.controller.scan_af_panel.channel_dropdown.value,
                                                   power=self.controller.scan_af_panel.power.value,
                                                   exposure=self.controller.scan_af_panel.set_exposure.value,
                                                   z_offset = [self.controller.scan_af_panel.zmin.value,self.controller.scan_af_panel.zmax.value])
                        best_z = z_list[np.argmax([img2freq(cv2_rescale(x,0.25)).mean() for x in scand])]
                        self.core.set_position(best_z)
                        self.core.wait_for_device(self.z_device)
                        group_z.append(best_z)
                        
                    fname = "{}/{}_{}Pos-{}_MDA_{}.tif".format(self.controller.filesave_pathselect.selected_path,
                                                      self.autosave_textpanel.value,well_name,pos,
                                                      str(self.counter).zfill(4))
                    metadata = tifffile_metadata_compiler(self.controller)
                    metadata['axes'] = axes
                    time_stamps = []
                    current_z = self.core.get_position()
                    position_z = []
                    canvas = np.zeros([len(self.z_scan_positions),len(channels),self.height,self.width])
                    for i,z in enumerate(self.z_scan_positions):
                        if stop_loop:
                            print("Experiment was manually terminated at {}".format(time_stamp()))
                            break
                        if z is not None:
                            z += current_z
                            position_z.append(z)
                            self.core.set_position(z)
                            self.core.wait_for_device(self.z_device)
                            for k,channel in enumerate(channels):
                                channel_name,light_src, expo, power = channel
                                metadata['C{}_name'.format(k+1)] = channel_name
                                metadata['C{}_LED_source'.format(k+1)] = light_src
                                metadata['C{}_exposure(ms)'.format(k+1)] = expo
                                metadata['C{}_LED_power%'.format(k+1)] = power
                                if light_src != '-':
                                    canvas[i,k] = capture_channel(self.led,
                                                                self.core,
                                                                channel=light_src,
                                                                exposure=expo,
                                                                power=power)
                                else:
                                    canvas[i,k] = capture_bright_field(self.core,exposure=expo)
                        else:
                            canvas = np.zeros([len(channels),self.height,self.width])
                            position_z.append(current_z)
                            for k,channel in enumerate(channels):
                                channel_name,light_src, expo, power = channel
                                metadata['C{}_name'.format(k+1)] = channel_name
                                metadata['C{}_LED_source'.format(k+1)] = light_src
                                metadata['C{}_exposure(ms)'.format(k+1)] = expo
                                metadata['C{}_LED_power%'.format(k+1)] = power
                                if light_src != '-':
                                    canvas[k] = capture_channel(self.led,
                                                                self.core,
                                                                channel=light_src,
                                                                exposure=expo,
                                                                power=power)
                                else:
                                    canvas[k] = capture_bright_field(self.core,exposure=expo)
                        time_stamps.append(time_stamp())
                    metadata['Z_positions'] = ';'.join([str(x) for x in position_z])
                    metadata['time_stamps'] = ';'.join(time_stamps)
                    self.core.set_position(current_z)
                    self.core.wait_for_device(self.z_device)
                    if stop_loop:
                        print("Experiment was manually terminated at {}".format(time_stamp()))
                        break
                    else:
                        tif.imwrite(fname,canvas.astype(np.uint16),metadata=metadata,imagej=True)
                    self.counter +=1
                if len(group_z)>1:
                    self.core.set_position(np.mean(group_z))
                    self.core.wait_for_device(self.z_device)
        if self.live_after_acq.value:
            self.core.set_shutter_open(True)
            self.controller.live_view_panel.umanager_live_snap.set_live_mode(True)
    
    def single_shot_snap(self):
        def on_click(change):
            self.controller.live_view_panel.umanager_live_snap.set_live_mode_on(False)
            metadata = tifffile_metadata_compiler(self.controller)
            metadata['axes'] = 'CYX'
            channels = self.controller.channel_panel.get_channel_info()
            fname = "{}/{}_SNAP_{}.tif".format(self.controller.filesave_pathselect.selected_path,
                                               self.autosave_textpanel.value,
                                               str(self.counter).zfill(4))
            canvas = np.zeros([len(channels),metadata['height'],metadata['width']])
            
            for k,channel in enumerate(channels):
                channel_name,light_src, expo, power = channel
                metadata['C{}_name'.format(k+1)] = channel_name
                metadata['C{}_LED_source'.format(k+1)] = light_src
                metadata['C{}_exposure(ms)'.format(k+1)] = expo
                metadata['C{}_LED_power%'.format(k+1)] = power
                if light_src != '-':
                    canvas[k] = capture_channel(self.led,
                                                self.core,
                                                channel=light_src,
                                                exposure=expo,
                                                power=power)
                else:
                    canvas[k] = capture_bright_field(self.core,exposure=expo)
                metadata['C{}_acquisition_time'.format(k+1)] = time_stamp()
            tif.imwrite(fname,canvas.astype(np.uint16),metadata=metadata,imagej=True)
            self.counter+=1
            if self.live_after_acq.value:
                self.core.set_shutter_open(True)
                self.controller.live_view_panel.umanager_live_snap.set_live_mode(True)
        return on_click

def iterative_auto_focus(TI_controller,
                         model,
                         deep_AF=False,
                         exposure=50,
                         zrange = [-3,0],
                         z_steps=4):
    mm_core = TI_controller.core
    TI_controller.live_view_panel.umanager_live_snap.set_live_mode_on(False)
    z_device = mm_core.get_focus_device()
    current_z = mm_core.get_position()
    rec_z = []
    if not deep_AF:
        for z in current_z + np.linspace(zrange[0],zrange[1],z_steps):
            mm_core.set_position(z)
            mm_core.wait_for_device(z_device)
            img=capture_bright_field(mm_core,exposure=exposure)
            offset = estimate_offset(img,model)
            rec_z.append(z-offset)
        mm_core.set_position(np.mean(rec_z))
        mm_core.wait_for_device(z_device)
    else:
        for i in range(z_steps):
            current_z = mm_core.get_position()
            img=capture_bright_field(mm_core,exposure=exposure)
            offset = DeepAF_predict(img,model)
            new_z = current_z-offset
            mm_core.set_position(new_z)
            mm_core.wait_for_device(z_device)

def DeepAF_predict(img,model,
                  max_offset = 18,
                  freq_norm_factor = 1500,
                  size=256,scale=0.25,):
    stack = img2patches(cv2_rescale(img,scale),size)
    freq = np.array([img2freq(x) for x in stack])
    freq = freq/freq_norm_factor
    freq[freq>1] = 1
    freq[freq<0] = 0
    pred = model.predict(freq,verbose=False)
    pred_norm = (pred*2-1)*max_offset
    return np.median(pred_norm)

def img2freq_raw(img):
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 50*np.log(np.abs(fshift))
    return magnitude_spectrum
    

def radio_sig(img):
    return radialaverage(img2freq(cv2_rescale(img,0.2)))

def AF_z_max_v(mm_core,mg_handler,
               light_src='UV',power=3,cutoff=1000,
              exposure=50, 
              num_z_slices = 7,
              z_offset = [-15,10]):
    
    z_device = mm_core.get_focus_device()
    camera = mm_core.get_camera_device()
    #快门常开，禁止自动快门
    mm_core.set_auto_shutter(False)
    mm_core.set_shutter_open(False)
    height = mm_core.get_image_height()
    width = mm_core.get_image_width()
    mm_core.set_exposure(float(exposure))
    z_list = []
    
    # 获取当前的Z轴空间位置
    z0 = mm_core.get_position()
    
    # 生成Z轴的相对位置
    z_relative = np.linspace(z_offset[0],z_offset[1],num_z_slices)
    
    #生成Z轴层扫的空间位置
    z_absolute = z0 + z_relative
    
    # 从下往上依次拍照
    
    canvas = np.zeros([num_z_slices,height,width])
    #"""
    
    for i,z in enumerate(z_absolute):
        mm_core.set_position(z)
        mm_core.wait_for_device(z_device)
        time.sleep(0.005)
        z_list.append(mm_core.get_position())
        
        if light_src != '-':
            canvas[i] = capture_channel(mg_handler,mm_core,channel=light_src,exposure=exposure,power=power)
            mm_core.wait_for_device(camera)
        else:
            canvas[i] = capture_bright_field(mm_core,exposure=exposure,shutter_off_after_acq=False)
    mm_core.set_shutter_open(False)
    mm_core.set_position(z0)
    mm_core.wait_for_device(z_device)
    return canvas, z_list
            

def tifffile_metadata_compiler(TI_controller):
    metadata = {}
    metadata['camera_pixel_size']=TI_controller.microscope_config_panel.camera_pixel_size
    metadata['magnification'] = TI_controller.microscope_config_panel.magnification
    metadata['pixel_microns'] = TI_controller.microscope_config_panel.pixel_microns
    metadata['width'] = TI_controller.md_acquisition_panel.width
    metadata['height'] = TI_controller.md_acquisition_panel.height
    metadata['camera_name'] = TI_controller.microscope_config_panel.camera
    metadata['objective_lens'] = TI_controller.microscope_config_panel.objective_lens
    metadata['NA'] = TI_controller.microscope_config_panel.na
    metadata['condensor'] = TI_controller.microscope_config_panel.condensor
    return metadata

def time_stamp():
    return str(datetime.datetime.now()).split('.')[0]


class Model_AF_Panel:

    def __init__(self):
        self.label = Label('Model-based auto-focus')
        self.set_exposure = BoundedFloatText(value=50,min=5,
                                               max=10000,
                                               step=1,
                                               description='Expo(ms)',
                                              disabled=False,layout=Layout(width='90%'))
        self.zmin = BoundedFloatText(value=-3,min=-15,max=0,step=0.1,description='Z scan lower',disabled=False,layout=Layout(width='90%'))
        self.zmax = BoundedFloatText(value=0,min=0,max=15,step=0.1,description='Z scan upper',disabled=False,layout=Layout(width='90%'))
        self.zsteps = BoundedIntText(value=3,min=1,max=11,description='# Steps',disabled=False,layout=Layout(width='90%'))
        self.layout = VBox([self.label,self.set_exposure,self.zmax,self.zmin,self.zsteps],layout=Layout(width='auto',border='1px dashed'))

class ZScan_AF_Panel:

    def __init__(self):
        self.label = Label('Iterative auto-focus')
        self.channel_dropdown = Dropdown(description='Light source ',
                                                   options=['UV','B','G','R','-'],
                                                   value='-',layout=Layout(width = '90%'))
        self.power = IntSlider(value=1,min=0,max=100,description='Power%',layout=Layout(width = '90%'))
        self.set_exposure = BoundedFloatText(value=50,min=5,
                                             max=10000,
                                             step=1,
                                             description='Expo(ms)',
                                             disabled=False,layout=Layout(width='90%'))
        self.zmin = BoundedFloatText(value=-1,min=-15,max=0,step=0.1,description='Z scan lower',disabled=False,layout=Layout(width='90%'))
        self.zmax = BoundedFloatText(value=1,min=0,max=15,step=0.1,description='Z scan upper',disabled=False,layout=Layout(width='90%'))
        self.zsteps = BoundedIntText(value=3,min=1,max=11,description='# Steps',disabled=False,layout=Layout(width='90%'))
        self.layout = VBox([self.label,self.channel_dropdown,self.power,
                            self.set_exposure,self.zmax,self.zmin,self.zsteps],layout=Layout(width='auto',border='1px dashed'))


class AdjustExposure:

    def __init__(self,mm_core):
        self.core = mm_core
        self.exposure = 50
        self.layout = BoundedFloatText(value=50,min=5,
                                       max=10000,
                                       step=1,
                                       description='Expo(ms)',
                                      disabled=False,layout=Layout(width='auto'))
        self.layout.observe(expo_control(self))

def expo_control(exposure_panel):
    def action(change):
        if exposure_panel.layout.value != exposure_panel.exposure:
            exposure_panel.core.set_exposure(exposure_panel.layout.value)
            exposure_panel.exposure = exposure_panel.layout.value
    return action


class MG_GUI:

    def __init__(self,mg_handler,mm_core):
        self.mg = mg_handler
        self.core = mm_core
        # 创建LED功率控制
        self.uv_slider = IntSlider(value=0,min=0,max=100,description='385nm')
        self.b_slider = IntSlider(value=0,min=0,max=100,description='460nm')
        self.g_slider = IntSlider(value=0,min=0,max=100,description='560nm')
        self.r_slider = IntSlider(value=0,min=0,max=100,description='625nm')
        self.bf_slider = IntSlider(value=0,min=0,max=100,description='-',disabled=True)
        
        # 创建LED开关按钮
        self.uv_button = create_button('UV')
        self.b_button = create_button('B')
        self.g_button = create_button('G')
        self.r_button = create_button('R')
        self.bf_button = create_button('BF')
        # 为每个按钮添加点击事件
        self.uv_button.on_click(led_control(self.uv_button, 'UV',self.uv_slider,self.mg))
        self.b_button.on_click(led_control(self.b_button, 'B',self.b_slider,self.mg))
        self.g_button.on_click(led_control(self.g_button, 'G',self.g_slider,self.mg))
        self.r_button.on_click(led_control(self.r_button, 'R',self.r_slider,self.mg))
        self.bf_button.on_click(illumination_control(self.bf_button,  self.core))
        
        uv_layer = HBox([self.uv_slider,self.uv_button],layout=Layout(width='auto'))
        b_layer = HBox([self.b_slider,self.b_button],layout=Layout(width='auto'))
        g_layer = HBox([self.g_slider,self.g_button],layout=Layout(width='auto'))
        r_layer = HBox([self.r_slider,self.r_button],layout=Layout(width='auto'))
        bf_layer = HBox([self.bf_slider,self.bf_button],layout=Layout(width='auto'))
        self.layout = VBox([uv_layer,b_layer,g_layer,r_layer,bf_layer],layout=Layout(border='1px solid',width='40%'))

# 创建按钮并设置初始颜色为灰色
def create_button(color):
    button = Button(
        description=color,
        disabled=False,
        button_style = '',
        style = {'button_color':'lightgrey'},
        tooltip='Click me',
        layout = Layout(width='20pix',border='1px dashed'))
    return button


def led_control(button,channel,slider,mg_handler):
    colors = {'UV': 'royalblue','B': 'lightgreen','G': 'orange','R': 'salmon',}
    def on_click(change):
        if button.style.button_color != colors[channel]:
            button.style.button_color = colors[channel]
            power = slider.value
            mg_handler.channel_power(channel=channel,power=slider.value)
            mg_handler.channel_on(channel=channel)
        else:
            button.style.button_color = 'lightgrey'
            mg_handler.channel_off(channel=channel)
    return on_click


def illumination_control(button,mm_core):
    def on_click(change):
        if not mm_core.get_shutter_open():
            button.style.button_color = 'white'
            mm_core.set_auto_shutter(False)
            mm_core.set_shutter_open(True)
        else:
            button.style.button_color = 'lightgrey'
            mm_core.set_auto_shutter(False)
            mm_core.set_shutter_open(False)
    return on_click



class ChannelSelectPanel:
    def __init__(self, initial_channels=1):
        self.default_config_path = "D:/Zhulab_microscopy_data/configs"
        self.configload_pathselect = FileChooser(self.default_config_path,
                                                 default_path=self.default_config_path,
                                                 select_default=False,
                                                 title='<b>Select channel configurations<b>',Layout = Layout(width='40%')) 
        self.save_config_button = Button(description='Save settings',Layout = Layout(width='15%'))
        self.load_config_button = Button(description='Load settings',Layout = Layout(width='15%'))
        self.channel_options = ['CY5', 'AF647', 'NucRed',
                                'CY3', 'TRITC', 'NileRed',
                                'FITC', 'AF488', 'NADA', 
                                'DAPI', 'HADA', 
                                'BrightField', 'PhaseContrast']
        self.LED_src = {'CY5':'R','AF647':'R','NucRed':'R',
                        'CY3':'G','TRITC':'G','NileRed':'G',
                        'FITC':'B','AF488':'B','NADA':'B',
                        'DAPI':'UV','HADA':'UV','BrightField':'-','PhaseContrast':'-'}
        self.n_channel_dropdown = Dropdown(description='# Channels',
                                                   options=np.arange(8).astype(int)+1,
                                                   value=initial_channels,
                                           layout=Layout(width = '25%'))
        self.channel_boxes = [ChannelBox(channel_index=i+1,channel_options=self.channel_options) for i in range(initial_channels)]
        self.layout = VBox(children=[HBox([self.n_channel_dropdown,
                                           VBox([self.save_config_button,self.load_config_button]),
                                           self.configload_pathselect])] + [box.layout for box in self.channel_boxes],
                           layout = Layout(border='1px solid',width = '70%'))
        self.n_channel_dropdown.observe(self.update_layout, names='value')
        self.save_config_button.on_click(channelconfig_control(self.save_config_button,self))
        self.load_config_button.on_click(load_config(self.load_config_button,self))
        
    def update_layout(self, change):
        current_num_boxes = len(self.channel_boxes)
        new_num_boxes = self.n_channel_dropdown.value
        if new_num_boxes > current_num_boxes:
            # 添加新的ChannelBox实例
            for i in range(new_num_boxes - current_num_boxes):
                new_box = ChannelBox(channel_index=current_num_boxes+i+1,channel_options=self.channel_options)
                self.channel_boxes.append(new_box)
                self.layout.children += (new_box.layout,)
        elif new_num_boxes < current_num_boxes:
            # 删除多余的ChannelBox实例
            for _ in range(current_num_boxes - new_num_boxes):
                self.channel_boxes.pop().layout.unobserve_all()
                self.layout.children = self.layout.children[:-1]


    def get_channel_info(self):
        channel_table = []
        for b in self.channel_boxes:
            channel = b.channel_dropdown.value
            exposure = b.exposure_panel.value
            power = b.power_panel.value
            channel_table.append([channel,self.LED_src[channel],exposure,power])
        return channel_table
        
def load_config(button,channel_select_panel):
    def on_click(change):
        if channel_select_panel.configload_pathselect.selected is not None:
            try:
                channel_info = pd.read_csv(channel_select_panel.configload_pathselect.selected)[['Channel', 'LED', 'Exposure','Power']].values
                channel_select_panel.n_channel_dropdown.value = len(channel_info)  
                while len(channel_select_panel.channel_boxes) > 0:
                    channel_select_panel.channel_boxes.pop().layout.unobserve_all()
                    channel_select_panel.layout.children = channel_select_panel.layout.children[:-1]
                
                # Add new ChannelBoxes based on loaded configuration
                for i, info in enumerate(channel_info, start=1):
                    channel_box = ChannelBox(channel_index=i, 
                                             channel_options=channel_select_panel.channel_options,channel_name=info[0],
                                             power = info[3],exposure=info[2])
                    channel_select_panel.channel_boxes.append(channel_box)
                    channel_select_panel.layout.children += (channel_box.layout,)
                    
            except Exception as e:
                print(f"Failed to load configuration from {channel_select_panel.configload_pathselect.selected}: {e}")
    return on_click
        

def channelconfig_control(button,channel_hander):
    def on_click(change):
        path = channel_hander.default_config_path
        channels = channel_hander.get_channel_info()
        dname = '_'.join(str(datetime.datetime.now()).split('.')[0].split(' ')).replace(':','-')
        fname = path+'/ChannelConfig_{}.csv'.format(dname)
        df = pd.DataFrame(channels,columns=['Channel', 'LED', 'Exposure','Power'])
        df.to_csv(fname)
    return on_click
    

class ChannelBox:
    def __init__(self,
                 channel_index=1,
                 channel_options=['CY5', 'AF647', 
                                'CY3', 'TRITC', 
                                'FITC', 'AF488', 'NADA', 
                                'DAPI', 'HADA', 
                                'BrightField', 'PhaseContrast'],
                 channel_name = None,
                 power=50,
                 exposure=50):
        self.channel_options = channel_options
        if channel_name is not None and channel_name in channel_options:
            self.channel_dropdown = Dropdown(description='Channel-{}'.format(channel_index),
                                             options=self.channel_options,
                                             value=channel_name,layout=Layout(width='30%'))
        else:
            self.channel_dropdown = Dropdown(description='Channel-{}'.format(channel_index),
                                                     options=self.channel_options,
                                                     value=self.channel_options[0],layout=Layout(width='30%'))
        self.exposure_panel = BoundedFloatText(value=exposure, min=5, max=10000.0, step=0.1, description='Expo(ms)',layout=Layout(width='25%'))
        self.power_panel = IntSlider(value=power, min=0, max=100, step=1, description='Power (%):', 
                                     continuous_update=False, orientation='horizontal', readout=True, readout_format='d',layout=Layout(width='45%'))
        self.layout = HBox(children=[self.channel_dropdown, self.exposure_panel, self.power_panel],
                           layout=Layout(width='auto'))


class ZStackPanel:

    def __init__(self):
        self.do_z_scan = Checkbox(value=False,description='Z Stack',disabled=False,indent=False,layout=Layout(width='auto'))
        self.up_scan = FloatSlider(value=2,min=0,max=20,step=0.1,
                                   description='Up',disabled=True,continuous_update=True,
                                   orientation='horizontal',readout=True,readout_format='.1f',
                                   layout=Layout(width='auto'))
        self.down_scan = FloatSlider(value=-2,min=-20,max=0,step=0.1,
                                     description='Down',disabled=True, continuous_update=True,
                                     orientation='horizontal',readout=True,readout_format='.1f',
                                     layout=Layout(width='auto'))

        def toggle_z_scan_options(change):
            self.up_scan.disabled = not change.new
            self.down_scan.disabled = not change.new
            self.scan_steps.disabled = not change.new

        # Attach the function to the checkbox
        self.do_z_scan.observe(toggle_z_scan_options, names='value')
        self.scan_steps = BoundedIntText(value=7,min=0,max=200,step=1,description='# Steps',disabled=True,layout=Layout(width='auto'))
        self.layout = VBox([self.do_z_scan,
                            self.up_scan,
                            self.down_scan,
                            self.scan_steps],layout=Layout(border='1px solid',width='35%'))


class TileScanPanel:

    def __init__(self):
        self.do_tilescan = Checkbox(value=False,description='Tile Scan',
                                    disabled=False,indent=False,layout=Layout(width='auto'))
        self.n_rows = IntSlider(value=4,min=1,max=50,step=1,
                                   description='# Rows',disabled=True,continuous_update=True,
                                   orientation='horizontal',readout=True,layout=Layout(width='auto'))
        self.n_cols = IntSlider(value=4,min=1,max=50,step=1,
                                   description='# Columns',disabled=True,continuous_update=True,
                                   orientation='horizontal',readout=True,layout=Layout(width='auto'))
        self.overlap = BoundedFloatText(value=-10,min=-300,max=50,step=0.1,description='% Overlap',disabled=False,layout=Layout(width='auto'))
        
        def toggle_tilescan_options(change):
            self.n_rows.disabled = not change.new
            self.n_cols.disabled = not change.new

        # Attach the function to the checkbox
        self.do_tilescan.observe(toggle_tilescan_options, names='value')
        self.layout = VBox([self.do_tilescan,self.n_rows,self.n_cols,self.overlap],
                           layout=Layout(border='1px solid',width='35%'))

class PlateScanPanel:

    def __init__(self,mm_core):
        self.core = mm_core
        self.options_rows = [2, 3, 4, 6, 8, 16]
        self.options_columns = [2, 3, 4, 6, 8, 12, 16, 24]
        self.n_rows = 4
        self.n_cols = 8
        self.layout = None
        self.top_left_coords = [1,0,0]
        self.top_right_coords = [1,1,0]
        self.bottom_left_coords = [0,0,0]
        self.bottom_right_coords = [0,1,0] 
        self.calibrated =False
        self.plate = None
        self.do_platescan = Checkbox(value=False,description='Plate Scan',
                                    disabled=False,indent=False,layout=Layout(width='auto'))
        self.use_magellan = Checkbox(value=True, description='Plate Magellan', 
                              disabled=False,indent=False,layout=Layout(width='auto'))
        self.dropdown_rows = Dropdown(options=self.options_rows,value=self.n_rows,description='N_rows:',layout=Layout(width='90%'),disabled=True)
        self.dropdown_cols = Dropdown(options=self.options_columns,value=self.n_cols,description='N_columns:',layout=Layout(width='90%'),disabled=True)
        self.button_layout_top_left = GetCoordsButton('Top-Left-Well:',self.top_left_coords,self.core)
        self.button_layout_top_right = GetCoordsButton('Top-Right-Well:',self.top_right_coords,self.core)
        self.button_layout_bottom_left = GetCoordsButton('Bottom-Left-Well:',self.bottom_left_coords,self.core)
        self.button_layout_bottom_right = GetCoordsButton('Bottom-Right-Well:',self.bottom_right_coords,self.core)
        self.update_plate_button = Button(value=False,
                                                 description='GO!',
                                                 disabled=True,
                                                 button_style='info', # 'success', 'info', 'warning', 'danger' or ''
                                                 tooltip='Description',
                                                 icon='check')
        self.plate_control_panel = Label('')
        self.magellan_model = None
        
        
        def toggle_platescan_options(change):
            self.dropdown_rows.disabled = not change.new
            self.dropdown_cols.disabled = not change.new
            self.button_layout_top_left.button.disabled = not change.new
            self.button_layout_top_right.button.disabled = not change.new
            self.button_layout_bottom_left.button.disabled = not change.new
            self.button_layout_bottom_right.button.disabled = not change.new
            self.update_plate_button.disabled = not change.new
        
        def update_ref_coords(change):
            self.top_left_coords = self.button_layout_top_left.xyz
            self.top_right_coords = self.button_layout_top_right.xyz
            self.bottom_left_coords = self.button_layout_bottom_left.xyz
            self.bottom_right_coords = self.button_layout_bottom_right.xyz

        def update_dropdown_values(change): 
            self.n_rows = self.dropdown_rows.value
            self.n_cols = self.dropdown_cols.value
                
        
        self.dropdown_rows.observe(update_dropdown_values, names='value')
        self.dropdown_cols.observe(update_dropdown_values, names='value')
        self.button_layout_top_left.button.on_click(update_ref_coords)
        self.button_layout_top_right.button.on_click(update_ref_coords)
        self.button_layout_bottom_left.button.on_click(update_ref_coords)
        self.button_layout_bottom_right.button.on_click(update_ref_coords)
        self.do_platescan.observe(toggle_platescan_options, names='value')
        
        self.layout = HBox([VBox([HBox([self.do_platescan,self.use_magellan]),
                                HBox([self.dropdown_rows,self.dropdown_cols]),
                                HBox([self.button_layout_top_left.layout,self.button_layout_top_right.layout]),
                                HBox([self.button_layout_bottom_left.layout,self.button_layout_bottom_right.layout]),
                                HBox([Label(value="Create Multi-Well Plate"),self.update_plate_button])],
                                layout = Layout(border='1px solid',width='30%')),
                                self.plate_control_panel.layout])

        def update_plate_layout(change):
            if self.update_plate_button.description=='GO!':
                self.update_plate_button.description='Start Over?'
                self.update_plate_button.icon = 'uncheck'
                well_tl = '{}{}'.format(chr(65+0),'01')
                well_tr = '{}{}'.format(chr(65+0),str(self.n_cols).zfill(2))
                well_bl = '{}{}'.format(chr(65+self.n_rows-1),'01')
                well_br = '{}{}'.format(chr(65+self.n_rows-1),str(self.n_cols).zfill(2))
                self.plate = PlateCoordsCalibration(n_rows=self.n_rows,n_cols=self.n_cols,method='affine')
                self.plate.calibrate({well_tl:self.top_left_coords,
                                      well_tr:self.top_right_coords,
                                      well_bl:self.bottom_left_coords,
                                      well_br:self.bottom_right_coords})
                self.plate_control_panel = MultiWellPlate(self,n_cols=self.n_cols,n_rows=self.n_rows)
                self.magellan_model = FlatPlaneFit(self.top_left_coords,
                                                  self.top_right_coords,
                                                    self.bottom_left_coords,
                                                    self.bottom_right_coords)
                
                self.layout.children = self.layout.children[:-1] + (self.plate_control_panel.layout,)
            else:
                self.update_plate_button.description='GO!'
                self.update_plate_button.icon = 'check'
                self.layout.children = self.layout.children[:-1] + (Label(''),)
        self.update_plate_button.on_click(update_plate_layout)

    def predict_z_magellan(self,x,y):
        if self.magellan_model is None:
            return 0
        else:
            return self.magellan_model.predict(x, y)
            
class MultiWellPlate:

    def __init__(self,plate_scan_panel,
                 n_rows=4,
                 n_cols=8,):
        self.parent = plate_scan_panel
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.button_dict = {}
        self.checkbox_dict = {}
        self.plate = plate_scan_panel.plate
        self.core = plate_scan_panel.core
        button_matrix=[]
        if max(self.n_rows,self.n_cols)<8:
            self.fontsize='14px'
        elif max(self.n_rows,self.n_cols)<12:
            self.fontsize='11px'
        else:
            self.fontsize='8px'
        for r in range(self.n_rows):
            row = []
            for c in range(self.n_cols):
                c_l = str(c+1).zfill(2)
                well = '{}{}'.format(chr(65+r),c_l)
                button = ToggleButton(value=False,
                                   description=well,
                                   disabled=False,
                                   button_style='success',
                                   style=dict(font_size=self.fontsize),
                                   indent=False, width='auto',height='auto')
                checkbox = Checkbox(value=True,description='',disabled=False,indent=False,layout = Layout(width='20%'))
                self.button_dict[well] = button
                self.checkbox_dict[well] = checkbox
                row.append(HBox([checkbox,button],layout = Layout(border='1px dashed')))
            button_matrix.append(HBox(row))
        self.layout = VBox(button_matrix,layout=Layout(border='1px solid',
                                                       width='58%',
                                                       height='auto'))
        
        def on_button_clicked(change):
            # 确保每次只有一个按钮被选中
            for k,b in self.button_dict.items():
                if b is not change.owner and b.value:
                    b.value = False
                elif b.value:
                    x,y = self.plate.get_well_coords(k)
                    self.core.set_xy_position(x,y)
                    self.core.wait_for_device(self.core.get_xy_stage_device())
                    if self.parent.use_magellan.value:
                        z = self.parent.magellan_model.predict(x,y)
                        self.core.set_position(z)
                        self.core.wait_for_device(self.core.get_focus_device())
        
        for k,b in self.button_dict.items():
            b.observe(on_button_clicked, names='value')

class PlateCoordsCalibration:

    def __init__(self,n_rows=8,n_cols=12,
                 method = 'warp'):

        self.n_rows=n_rows
        self.n_cols=n_cols
        well_names,well_coords = initiate_multi_well_plate(self.n_rows,self.n_cols)
        self.well_names = well_names
        self._well2id = {x:i for i,x in enumerate(well_names)}
        self._ref_coords_all = well_coords
        self._ref_coords_calib = None
        self.coords_all = None
        self.coords_calib = None
        self.affine_matrix = None
        self.method = method
        
    def calibrate(self,coords_dict = {'A01':(46740.1,34265.7,0),
                                   'H01':(46830.8,-29378.7,0),
                                   'A12':(-52145.0,33638.6,0),
                                   'H12':(-51685.8,-29193.2,0)}):
        ref_coords_calib = []
        coords_calib = []
        for w,coords in coords_dict.items():
            if w not in self.well_names:
                raise ValueError('Invalid well name - {} - found!'.format(w))
            else:
                ref_coords_calib.append(self._ref_coords_all[self._well2id[w]])
                coords_calib.append(coords[:2])
        ref_coords_calib = np.array(ref_coords_calib).astype(np.float32)
        coords_calib = np.array(coords_calib).astype(np.float32)
        if self.method not in ['warp','affine']:
            raise ValueError('Invalid method {}! Should be either "warp" or "affine".'.format(self.method))
        if self.method == 'affine':
            self.affine_matrix = cv2.estimateAffinePartial2D(ref_coords_calib,coords_calib)[0]
            self.coords_calib = cv2.transform(np.array([self._ref_coords_all]),self.affine_matrix)[0]
        else:
            self.warp_matrix = cv2.getPerspectiveTransform(ref_coords_calib,coords_calib)
            self.coords_calib = cv2.transform(np.array([self._ref_coords_all]),self.warp_matrix)[0][:,:2]
        return self

    def get_well_coords(self,well='A03'):
        if well not in self.well_names:
            raise ValueError('Invalid well name - {}!'.format(well))
        return self.coords_calib[self._well2id[well]].astype(float)


        
def initiate_multi_well_plate(n_rows=8,
                              n_cols=12):
    well_names = []
    well_coords = []
    for r in range(n_rows):
        for c in range(n_cols):
            wname = '{}{}'.format(chr(r+65),str(c+1).zfill(2))
            well_names.append(wname)
            well_coords.append([c,r])
    return well_names, np.array(well_coords,dtype=np.float32)

class GetCoordsButton:

    def __init__(self,
                 label,
                 assign_to,
                 mm_core):
        self.xyz = assign_to
        self.button = Button(description=label,
                             disabled=True)
        self.text_x = BoundedFloatText(value=self.xyz[0],min=-1000000,max=1000000, step=0.1,description='X',disabled=True,layout=Layout(width='95%'))
        self.text_y = BoundedFloatText(value=self.xyz[1],min=-1000000,max=1000000, step=0.1,description='Y',disabled=True,layout=Layout(width='95%'))
        self.core = mm_core

        def update_position(change):
            current_x = self.core.get_x_position()
            current_y = self.core.get_y_position()
            current_z = self.core.get_position()
            self.text_x.value = current_x
            self.text_y.value = current_y
            self.xyz = [current_x,current_y,current_z]
            
        self.button.on_click(update_position)
        self.layout = VBox([self.button,self.text_x,self.text_y])

    def freeze(self):
        self.button.disabled=True
        self.text_x.disabled=True
        self.text_y.disabled=True

class LiveViewButton:

    def __init__(self,mm_studio,mm_core,label='Live',auto_bf = False):
        self.studio = mm_studio
        self.core = mm_core
        self.umanager_displays = self.studio.displays()
        self.umanager_live_snap = self.studio.get_snap_live_manager()
        self.live_button = create_button(label)
        self.live_button.on_click(live_button_control(self.live_button,
                                                      self.umanager_live_snap,
                                                      self.core,auto_bf))


def live_button_control(button,
                        live_snap_controller,
                        mm_core,
                        auto_bf):
    def on_click(change):
        if live_snap_controller.get_is_live_mode_on():
            live_snap_controller.set_live_mode(False)
            if auto_bf:
                mm_core.set_auto_shutter(False)
                mm_core.set_shutter_open(False)
                mm_core.wait_for_device(mm_core.get_shutter_device())
            button.style.button_color = 'lightgrey'
        else:
            button.style.button_color = 'lightgreen'
            if auto_bf:
                mm_core.set_auto_shutter(False)
                mm_core.set_shutter_open(True)
                mm_core.wait_for_device(mm_core.get_shutter_device())
            live_snap_controller.set_live_mode(True)
    return on_click
    

def AF_z_scanning(mm_core,mg_handler,light_src='G',power=5,
                  exposure=50, 
                  num_z_slices = 5,
                  z_offset = [-3,3]):
    
    z_device = mm_core.get_focus_device()
    camera = mm_core.get_camera_device()
    #快门常开，禁止自动快门
    mm_core.set_auto_shutter(False)
    mm_core.set_shutter_open(False)
    height = mm_core.get_image_height()
    width = mm_core.get_image_width()
    mm_core.set_exposure(float(exposure))
    z_list = []
    
    # 获取当前的Z轴空间位置
    z0 = mm_core.get_position()
    
    # 生成Z轴的相对位置
    z_relative = np.linspace(z_offset[0],z_offset[1],num_z_slices)
    
    #生成Z轴层扫的空间位置
    z_absolute = z0 + z_relative
    
    # 从下往上依次拍照
    
    canvas = np.zeros([num_z_slices,height,width])
    #"""
    
    for i,z in enumerate(z_absolute):
        mm_core.set_position(z)
        mm_core.wait_for_device(z_device)
        time.sleep(0.005)
        z_list.append(mm_core.get_position())
        
        if light_src != '-':
            canvas[i] = capture_channel(mg_handler,mm_core,channel=light_src,exposure=exposure,power=power)
            mm_core.wait_for_device(camera)
        else:
            canvas[i] = capture_bright_field(mm_core,exposure=exposure)
    mm_core.set_position(z0)
    mm_core.wait_for_device(z_device)
    return canvas, z_list


def estimate_offset(image,model,scale=0.2,preserve_range=False, use_radial=True):
    from skimage import transform
    return model.predict(img2freq_fullstats(image,scale=scale,
                                            preserve_range=preserve_range,
                                            use_radial=use_radial).reshape(1,-1))[0]


def img2freq_fullstats(img,scale=0.2,preserve_range=False,
                       use_radial=True):
    rescaled = transform.rescale(img,scale,preserve_range=preserve_range)
    f = np.fft.fft2(rescaled)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 50*np.log(np.abs(fshift))
    f_stat = magnitude_spectrum.mean(axis=0)
    if use_radial:
        f_stat = np.hstack([f_stat,radialaverage(magnitude_spectrum)])
    return f_stat
    
def radialaverage(image):
    image_shape = image.shape
    center = (image_shape[0] / 2, image_shape[1] / 2)
    #index pixels based on which bin they fall into
    dx, dy = np.meshgrid(np.arange(image_shape[0]) - center[0], np.arange(image_shape[1]) - center[1])
    dist = np.sqrt(dx ** 2 + dy ** 2)
    integerdist = np.round(dist).astype(int)
    maxdist = min(integerdist[:,-1]) #bottom edge is closer to center pixel than top
    radialavg = np.zeros(maxdist)
    for i in np.arange(maxdist):
        radialavg[i] = np.mean(image[integerdist==i])
    return radialavg

def capture_channel(mg_handler,
                    mm_core,
                    exposure=50,
                    channel='UV',
                    power=10,
                    pre_expo_lag=100,
                    post_expo_lag=50):
    height = mm_core.get_image_height()
    width = mm_core.get_image_width()
    shutter_device = mm_core.get_shutter_device()
    camera = mm_core.get_camera_device()
    
    mm_core.set_auto_shutter(False)
    mm_core.set_shutter_open(False)
    mm_core.wait_for_device(shutter_device)
    
    mm_core.set_exposure(exposure)
    mg_handler.channel_power(channel,power=power)
    stop_event, _ = start_timed_task(mg_handler,
                                     channel,expo=exposure,pre_expo_lag=pre_expo_lag,
                                     post_expo_lag=post_expo_lag)
    mm_core.snap_image()
    
    mm_core.wait_for_device(camera)
    stop_event.set()
    return mm_core.get_image().reshape((height,width))


def capture_bright_field(mm_core,
                         exposure=200,
                         shutter_off_after_acq=True):
    height = mm_core.get_image_height()
    width = mm_core.get_image_width()
    camera = mm_core.get_camera_device()
    shutter = mm_core.get_shutter_device()
    mm_core.set_auto_shutter(False)
    mm_core.set_shutter_open(True)
    mm_core.wait_for_device(shutter)
    mm_core.set_exposure(float(exposure))
    mm_core.snap_image()
    mm_core.wait_for_device(camera)
    if shutter_off_after_acq:
        mm_core.set_shutter_open(False)
        mm_core.wait_for_device(shutter)
    return mm_core.get_image().reshape((height,width))

def generate_xy_rel(n_rows = 3,n_cols=3,step=216):
    dxy = []
    for _r in range(n_rows):
        r = _r-int(n_rows/2)
        for _c in range(n_cols):
            if _r%2==1:
                c = n_cols-1-_c
                
            else:
                c = _c
            c -= int(n_cols/2)
            dxy.append([r*step,c*step])
    return dxy

def timed_task(mg, 
               stop_event, 
               channel='B',
               pre_expo_lag=85, 
               expo=50,
               post_expo_lag=30):
    time.sleep(pre_expo_lag/1000)
    mg.channel_on(channel=channel)
    if stop_event.is_set():
        mg.channel_off(channel)
    time.sleep((expo+post_expo_lag)/1000)
    mg.channel_off(channel)

def start_timed_task(mg, 
                     channel='B',
                     pre_expo_lag=65, 
                     expo=50,
                     post_expo_lag=30):
    stop_event = threading.Event()
    thread = threading.Thread(target=timed_task, args=(mg, 
                                                       stop_event, 
                                                       channel, 
                                                       pre_expo_lag, 
                                                       expo,
                                                       post_expo_lag))
    thread.start()
    return stop_event, thread


def is_inside(point, polygon):
    return Path(polygon).contains_point(point)


class XYZ_recorder:

    def __init__(self,core,verbose=True):
        self.core = core
        self.coords = []
        self.verbose = verbose
        self.record_button = create_button('record')
        self.restart = create_button('restart')
        self.delete_last = create_button('del previous')
        
        def _record_click(change):
            self._record()

        def _restart_click(change):
            self.coords = []

        def _delete_previous(change):
            if len(self.coords)>0:
                self.coords = self.coords[:-1]
                if self.verbose:
                    print(self.coords)
            
        self.record_button.on_click(_record_click)
        self.delete_last.on_click(_delete_previous)
        self.restart.on_click(_restart_click)
        display(HBox([self.record_button,self.delete_last,self.restart]))
        
    def _record(self,verbose=False):
        self.coords.append(self._get_xyz_coords())
        if self.verbose:
            print(self.coords)

    def save_coords(self,filename):
        np.save(filename,np.array(self.coords))
                           
    def _get_xyz_coords(self):
        x=self.core.get_x_position()
        y=self.core.get_y_position()
        z=self.core.get_position()
        return [x,y,z]

    
def cv2_rescale(img,scale=0.5):
    # use opencv to resize an image which is substantially faster than skimage
    new_shape = np.flip((np.array(img.shape)*scale).astype(int))
    return cv2.resize(img,new_shape,cv2.INTER_LINEAR)


def img2patches(img,size=256):
    patchifier = Patchifier(img_shape=img.shape,patch_size=size,pad=0)
    images=patchifier.pachify(img)
    return images


def img2freq(img):
    # Convert the image to a tensor if it's not already
    img_tensor = tf.convert_to_tensor(img, dtype=tf.complex64)
    
    # Compute the 2D FFT
    f = tf.signal.fft2d(img_tensor)
    
    # Shift the zero-frequency component to the center
    fshift = tf.signal.fftshift(f)
    
    # Compute the magnitude spectrum
    magnitude_spectrum = 50 * tf.math.log(tf.abs(fshift))
    
    # Convert the result to a numpy array and return
    return magnitude_spectrum.numpy()


def axial_profile(magnitude_spectrum):
    rows, cols = magnitude_spectrum.shape
    center = (rows // 2, cols // 2)
    
    # Create coordinate grids
    y, x = np.indices((rows, cols))
    y = y - center[0]
    x = x - center[1]
    
    # Convert Cartesian coordinates to polar coordinates
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    
    # Sort radii and magnitude spectrum
    r_flat = r.flatten()
    magnitude_spectrum_flat = magnitude_spectrum.flatten()
    
    # Sort by radius
    sorted_indices = np.argsort(r_flat)
    r_sorted = r_flat[sorted_indices]
    magnitude_spectrum_sorted = magnitude_spectrum_flat[sorted_indices]
    
    # Bin the values by radius
    r_bin_edges = np.arange(0, np.max(r_sorted) + 1)
    r_bin_centers = (r_bin_edges[:-1] + r_bin_edges[1:]) / 2
    radial_profile = np.zeros_like(r_bin_centers)
    
    # Compute the radial profile by averaging the values within each radius bin
    for i in range(len(r_bin_centers)):
        bin_mask = (r_sorted >= r_bin_edges[i]) & (r_sorted < r_bin_edges[i + 1])
        if np.sum(bin_mask) > 0:
            radial_profile[i] = np.mean(magnitude_spectrum_sorted[bin_mask])
        else:
            radial_profile[i] = 0
    
    return r_bin_centers, radial_profile

class Patchifier:
    """
    A simple way to convert 2D images to patches and stitch them back into one
    Currently it only works with images with shapes like (height, width) or (height, width, channel), it doesn't work
    on image series such as (frame, height, width, channel)
    The smoothing function for overlap edges is simply the mean values of the overlapping pixels. Future updates may
    consider implementing 2D spline interpolation based smoothing method, such as:
    https://github.com/bnsreenu/python_for_microscopists/blob/master/229_smooth_predictions_by_blending_patches/smooth_tiled_predictions.py
    """

    def __init__(self, 
                 img_shape=(512, 512), 
                 patch_size=256, pad=32):

        """
        :param img_shape: the original shape of the large input image
        :param patch_size: the height and width of the square-shaped patch
        :param pad: half-width of the overlapping region of neighboring patches
        """

        self._shape = img_shape
        self.size = patch_size
        self.pad = pad
        self.shape = (max(self._shape[0], self.size),
                      max(self._shape[1], self.size))
        self.pad_h = max(0, self.size - self._shape[0])
        self.pad_w = max(0, self.size - self._shape[1])
        self.ref_coords = self.generate_patch_coords()

    def generate_patch_coords(self):
        """
        funtion to generate patch coords
        :return:
        """
        h, w = self.shape
        xs = list(np.arange(0, h - self.size, self.size - 2 * self.pad)) + [h - self.size]
        if len(xs) > 1:
            if xs[-1] == xs[-2]:
                xs = xs[:-1]
        ys = list(np.arange(0, w - self.size, self.size - 2 * self.pad)) + [w - self.size]
        if len(ys) > 1:
            if ys[-1] == ys[-2]:
                ys = ys[:-1]
        ref_coords = np.array([[x, y, np.random.randint(2)] for x in xs for y in ys])
        return ref_coords

    def pachify(self, img, random_rotate=False):
        """
        convert img to patches
        :param img: momia2 image
        :param random_rotate: if randomly rotate clips, this shouldn't be used for prediction but can be helpful for training
        :return: clipped patches
        """
        if self.shape != img.shape[:2]:
            self.__init__(img.shape[:2])
        pad_config = np.zeros((len(img.shape), 2))
        pad_config[0][1] = self.pad_h
        pad_config[1][1] = self.pad_w
        pad_config = pad_config.astype(int)
        if self.pad_h > 0 or self.pad_w > 0:
            padded_img = np.pad(img.copy(), pad_config, mode='constant')
        else:
            padded_img = img.copy()
        patches = []
        for x, y, t in self.ref_coords:
            p = padded_img[x:x + self.size, y:y + self.size]
            if random_rotate and t:
                p = p.T
            patches.append(p)
        return np.array(patches)

    def unpatchify(self, patches, n_channel):
        """
        stitch patches back into one
        :param patches: array of patches
        :param n_channel: number of channels, for instance, for a rgb image n_channel should be 3
        :return:
        """
        canvas = np.zeros(list(self.shape) + [n_channel])
        canvas_counter = np.zeros(self.shape)
        for i, p in enumerate(patches):
            x, y = self.ref_coords[i][0], self.ref_coords[i][1]
            canvas[x:x + self.size, y:y + self.size] += p
            canvas_counter[x:x + self.size, y:y + self.size] += 1
        mean_canvas = canvas / canvas_counter[:, :, np.newaxis]
        return mean_canvas[:self._shape[0], :self._shape[1]]

    def unpatchify_max(self, patches, n_channel):
        """
        stitch patches back into one
        :param patches: array of patches
        :param n_channel: number of channels, for instance, for a rgb image n_channel should be 3
        :return:
        """
        canvas = np.zeros(list(self.shape) + [n_channel])
        #canvas_counter = np.zeros(self.shape)
        for i, p in enumerate(patches):
            x, y = self.ref_coords[i][0], self.ref_coords[i][1]
            p0 = canvas[x:x + self.size, y:y + self.size].copy()
            max_p = np.array([p,p0]).max(axis=0)
            canvas[x:x + self.size, y:y + self.size] = max_p
            #canvas_counter[x:x + self.size, y:y + self.size] += max_p
        #mean_canvas = canvas / canvas_counter[:, :, np.newaxis]
        return canvas[:self._shape[0], :self._shape[1]]


from tensorflow.keras import layers
def ConvBlock(x, filter_size, filter_num, dropout=0, batch_norm=True):
    """
    inherited from @DigitalSreeni
    Standard 2xconvolution block inherited from DigitalSreeni
    :params x: input tensor
    :params filter_size: size of the square 2D convolution filters, goes by (filter_size, filter_size))
    :params filter_num: number of convolution filters
    :params dropout: dropout layer, default is 0 (no dropout)
    :params batch_norm: batch normalization layer
    :return: convolution block
    """
    # convolution layer
    conv = layers.Conv2D(filter_num, (filter_size,filter_size), padding='same')(x)
    # batch normalization
    if batch_norm:
        conv = layers.BatchNormalization(axis=3)(conv)
    conv = layers.Activation('relu')(conv)
    
    # consequtive conv layer
    # convolution layer
    conv = layers.Conv2D(filter_num, (filter_size,filter_size), padding='same')(conv)
    # batch normalization
    if batch_norm:
        conv = layers.BatchNormalization(axis=3)(conv)
    conv = layers.Activation('relu')(conv)
    
    if dropout > 0:
        conv = layers.Dropout(dropout)(conv)
    return conv

def CNN(input_shape, 
         n_classes=1, 
         filter_num=8,
         filter_size=3,
         dropout_rate=0.05,
         activation='sigmoid',
         batch_norm=True):
    """
    inherited from @DigitalSreeni
    Standard UNet, with attention 
    :params n_classes: number of output classes
    :params filter_num: number of basic filters for the first layer
    :params filter_size: size of the convolutional filter
    :params dropout_rate: dropout rate
    :params batch_norm: batch normalization, default is True
    :params up_sampling_size: size of upsampling filters
    :params activation: activation function
    :return: gating feature map with the same dimension of the up layer feature map
    """
    
    # input data
    # dimension of the image depth
    inputs = layers.Input(input_shape, dtype=tf.float32)
    axis = 3

    # Downsampling layers
    # Down ResConv 1, double residual convolution + pooling
    conv_1 = ConvBlock(inputs, filter_size, filter_num, 
                       dropout_rate, 
                       batch_norm)
    pool_1 = layers.MaxPooling2D(pool_size=(2,2))(conv_1)
    # Down ResConv 2, double residual convolution + pooling
    conv_2 = ConvBlock(pool_1, filter_size, 2*filter_num, dropout_rate, batch_norm)
    pool_2 = layers.MaxPooling2D(pool_size=(2,2))(conv_2)
    # Down ResConv 3, double residual convolution + pooling
    conv_3 = ConvBlock(pool_2, filter_size, 4*filter_num, dropout_rate, batch_norm)
    pool_3 = layers.MaxPooling2D(pool_size=(2,2))(conv_3)
    # Down ResConv 4, double residual convolution + pooling
    conv_4 = ConvBlock(pool_3, filter_size, 8*filter_num, dropout_rate, batch_norm)
    pool_4 = layers.MaxPooling2D(pool_size=(2,2))(conv_4)
    # Down ResConv 5, convolution only
    conv_5 = ConvBlock(pool_4, filter_size, 16*filter_num, dropout_rate, batch_norm)
    pool_4 = layers.MaxPooling2D(pool_size=(2,2))(conv_5)
    flat = layers.Flatten()(pool_4)
    dense = layers.Dense(64, activation='relu')(flat)
    dense_norm = layers.BatchNormalization()(dense)
    output = layers.Dense(1,activation=activation)(dense_norm)  # 输出层，只有一个神经元，输出0-1之间的值
    model = keras.models.Model(inputs, output, name="CNN")
    return model


class FlatPlaneFit:

    def __init__(self,p1,p2,p3,p4):
        self.A = np.array([[p1[0], p1[1], 1],
                          [p2[0], p2[1], 1],
                          [p3[0], p3[1], 1],
                          [p4[0], p4[1], 1]])
        self.b = np.array([p1[2], p2[2], p3[2], p4[2]])
        self.coeffs = np.linalg.lstsq(self.A, self.b, rcond=None)[0]

    def predict(self,x,y):
        a, b, c = self.coeffs
        return a * x + b * y + c

def fit_smooth_curvep(p1,p2,p3,p4):
    # 四个点的坐标
    points = np.array([p1,p2,p3,p4])
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    # 使用径向基函数（RBF）进行插值
    rbf = Rbf(x, y, z)
    return rbf

