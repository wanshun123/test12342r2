import traceback
import os
import sys
import time
import multiprocessing
from tqdm import tqdm
from pathlib import Path
import numpy as np
import cv2
from utils import Path_utils
from utils.DFLPNG import DFLPNG
from utils import image_utils
from facelib import FaceType
import facelib
from nnlib import nnlib

#from PIL import Image
#from mtcnn.mtcnn import MTCNN


from utils.SubprocessorBase import SubprocessorBase
class ExtractSubprocessor(SubprocessorBase):

    #override
    def __init__(self, input_data, type, image_size, face_type, debug, own_video, multi_gpu=False, cpu_only=False, manual=False, manual_window_size=0, detector=None, output_path=None ):
        self.input_data = input_data
        self.type = type
        self.image_size = image_size
        self.face_type = face_type
        self.debug = debug
        self.own_video = own_video
        self.multi_gpu = multi_gpu
        self.cpu_only = cpu_only
        self.detector = detector
        self.output_path = output_path
        self.manual = manual
        self.manual_window_size = manual_window_size
        self.result = []

        self.count = 1

        no_response_time_sec = 60 if not self.manual else 999999
        super().__init__('Extractor', no_response_time_sec)

    #override
    def onHostClientsInitialized(self):
        if self.manual == True:
            self.wnd_name = 'Manual pass'
            cv2.namedWindow(self.wnd_name)

            self.landmarks = None
            self.param_x = -1
            self.param_y = -1
            self.param_rect_size = -1
            self.param = {'x': 0, 'y': 0, 'rect_size' : 5, 'rect_locked' : False, 'redraw_needed' : False }

            def onMouse(event, x, y, flags, param):
                if event == cv2.EVENT_MOUSEWHEEL:
                    mod = 1 if flags > 0 else -1
                    param['rect_size'] = max (5, param['rect_size'] + 10*mod)
                elif event == cv2.EVENT_LBUTTONDOWN:
                    param['rect_locked'] = not param['rect_locked']
                    param['redraw_needed'] = True
                elif not param['rect_locked']:
                    param['x'] = x
                    param['y'] = y

            cv2.setMouseCallback(self.wnd_name, onMouse, self.param)

    def get_devices_for_type (self, type, multi_gpu):
        if (type == 'rects' or type == 'landmarks'):
            if not multi_gpu:
                devices = [nnlib.device.getBestDeviceIdx()]
            else:
                devices = nnlib.device.getDevicesWithAtLeastTotalMemoryGB(2)
            devices = [ (idx, nnlib.device.getDeviceName(idx), nnlib.device.getDeviceVRAMTotalGb(idx) ) for idx in devices]

        elif type == 'final':
            devices = [ (i, 'CPU%d' % (i), 0 ) for i in range(0, multiprocessing.cpu_count()) ]

        return devices

    #override
    def process_info_generator(self):
        base_dict = {'type' : self.type,
                     'image_size': self.image_size,
                     'face_type': self.face_type,
                     'debug': self.debug,
                     'output_dir': str(self.output_path),
                     'detector': self.detector}

        if self.cpu_only:
            num_processes = 1
            if not self.manual and self.type == 'rects' and self.detector == 'mt':
                num_processes = int ( max (1, multiprocessing.cpu_count() / 2 ) )

            for i in range(0, num_processes ):
                client_dict = base_dict.copy()
                client_dict['device_idx'] = 0
                client_dict['device_name'] = 'CPU' if num_processes == 1 else 'CPU #%d' % (i),
                client_dict['device_type'] = 'CPU'

                yield client_dict['device_name'], {}, client_dict

        else:
            for (device_idx, device_name, device_total_vram_gb) in self.get_devices_for_type(self.type, self.multi_gpu):
                num_processes = 1
                if not self.manual and self.type == 'rects' and self.detector == 'mt':
                    num_processes = int ( max (1, device_total_vram_gb / 2) )

                for i in range(0, num_processes ):
                    client_dict = base_dict.copy()
                    client_dict['device_idx'] = device_idx
                    client_dict['device_name'] = device_name if num_processes == 1 else '%s #%d' % (device_name,i)
                    client_dict['device_type'] = 'GPU'

                    yield client_dict['device_name'], {}, client_dict

    #override
    def get_no_process_started_message(self):
        if (self.type == 'rects' or self.type == 'landmarks'):
            print ( 'You have no capable GPUs. Try to close programs which can consume VRAM, and run again.')
        elif self.type == 'final':
            print ( 'Unable to start CPU processes.')

    #override
    def onHostGetProgressBarDesc(self):
        return None

    #override
    def onHostGetProgressBarLen(self):
        return len (self.input_data)

    #override
    def onHostGetData(self):
        if not self.manual:
            if len (self.input_data) > 0:
                return self.input_data.pop(0)
        else:
            skip_remaining = False
            allow_remark_faces = False
            while len (self.input_data) > 0:
                data = self.input_data[0]
                filename, faces = data
                is_frame_done = False
                go_to_prev_frame = False

                # Can we mark an image that already has a marked face?
                if allow_remark_faces:
                    allow_remark_faces = False
                    # If there was already a face then lock the rectangle to it until the mouse is clicked
                    if len(faces) > 0:
                        prev_rect = faces.pop()[0]
                        self.param['rect_locked'] = True
                        faces.clear()
                        self.param['rect_size'] = ( prev_rect[2] - prev_rect[0] ) / 2
                        self.param['x'] = ( ( prev_rect[0] + prev_rect[2] ) / 2 ) * self.view_scale
                        self.param['y'] = ( ( prev_rect[1] + prev_rect[3] ) / 2 ) * self.view_scale

                if len(faces) == 0:
                    self.original_image = cv2.imread(filename)

                    (h,w,c) = self.original_image.shape

                    self.view_scale = 1.0 if self.manual_window_size == 0 else self.manual_window_size / (w if w > h else h)
                    self.original_image = cv2.resize (self.original_image, ( int(w*self.view_scale), int(h*self.view_scale) ), interpolation=cv2.INTER_LINEAR)
                    (h,w,c) = self.original_image.shape

                    self.text_lines_img = (image_utils.get_draw_text_lines ( self.original_image, (0,0, self.original_image.shape[1], min(100, self.original_image.shape[0]) ),
                                                    [   'Match landmarks with face exactly. Click to confirm/unconfirm selection',
                                                        '[Enter] - confirm and continue to next unmarked frame',
                                                        '[Space] - skip to next unmarked frame',
                                                        '[Mouse wheel] - change rect',
                                                        '[,] [.]- prev frame, next frame',
                                                        '[Q] - skip remaining frames'
                                                    ], (1, 1, 1) )*255).astype(np.uint8)

                    while True:
                        key = cv2.waitKey(1) & 0xFF

                        if key == ord('\r') or key == ord('\n'):
                            faces.append ( [(self.rect), self.landmarks] )
                            is_frame_done = True
                            break
                        elif key == ord(' '):
                            is_frame_done = True
                            break
                        elif key == ord('.'):
                            allow_remark_faces = True
                            # Only save the face if the rect is still locked
                            if self.param['rect_locked']:
                                faces.append ( [(self.rect), self.landmarks] )
                            is_frame_done = True
                            break
                        elif key == ord(',')  and len(self.result) > 0:
                            # Only save the face if the rect is still locked
                            if self.param['rect_locked']:
                                faces.append ( [(self.rect), self.landmarks] )
                            go_to_prev_frame = True
                            break
                        elif key == ord('q'):
                            skip_remaining = True
                            break

                        new_param_x = self.param['x'] / self.view_scale
                        new_param_y = self.param['y'] / self.view_scale
                        new_param_rect_size = self.param['rect_size']

                        new_param_x = np.clip (new_param_x, 0, w-1)
                        new_param_y = np.clip (new_param_y, 0, h-1)

                        if self.param_x != new_param_x or \
                           self.param_y != new_param_y or \
                           self.param_rect_size != new_param_rect_size or \
                           self.param['redraw_needed']:

                            self.param_x = new_param_x
                            self.param_y = new_param_y
                            self.param_rect_size = new_param_rect_size

                            self.rect = (self.param_x-self.param_rect_size, self.param_y-self.param_rect_size, self.param_x+self.param_rect_size, self.param_y+self.param_rect_size)
                            return [filename, [self.rect]]

                else:
                    is_frame_done = True

                if is_frame_done:
                    self.result.append ( data )
                    self.input_data.pop(0)
                    self.inc_progress_bar(1)
                    self.param['redraw_needed'] = True
                    self.param['rect_locked'] = False
                elif go_to_prev_frame:
                    self.input_data.insert(0, self.result.pop() )
                    self.inc_progress_bar(-1)
                    allow_remark_faces = True
                    self.param['redraw_needed'] = True
                    self.param['rect_locked'] = False
                elif skip_remaining:
                    while len(self.input_data) > 0:
                        self.result.append( self.input_data.pop(0) )
                        self.inc_progress_bar(1)

        return None

    #override
    def onHostDataReturn (self, data):
        if not self.manual:
            self.input_data.insert(0, data)

    #override
    def onClientInitialize(self, client_dict):
        self.safe_print ('Running on %s.' % (client_dict['device_name']) )
        self.type         = client_dict['type']
        self.image_size   = client_dict['image_size']
        self.face_type    = client_dict['face_type']
        self.device_idx   = client_dict['device_idx']
        self.cpu_only     = client_dict['device_type'] == 'CPU'
        self.output_path  = Path(client_dict['output_dir']) if 'output_dir' in client_dict.keys() else None
        self.debug        = client_dict['debug']
        self.detector     = client_dict['detector']

        self.e = None

        device_config = nnlib.DeviceConfig ( cpu_only=self.cpu_only, force_best_gpu_idx=self.device_idx, allow_growth=True)
        if self.type == 'rects':
            if self.detector is not None:
                if self.detector == 'mt':
                    nnlib.import_all (device_config)
                    self.e = facelib.MTCExtractor(nnlib.keras, nnlib.tf, nnlib.tf_sess)
                elif self.detector == 'dlib':
                    nnlib.import_dlib (device_config)
                    self.e = facelib.DLIBExtractor(nnlib.dlib)
                self.e.__enter__()

        elif self.type == 'landmarks':
            nnlib.import_all (device_config)
            self.e = facelib.LandmarksExtractor(nnlib.keras)
            self.e.__enter__()

        elif self.type == 'final':
            pass

        return None

    #override
    def onClientFinalize(self):
        if self.e is not None:
            self.e.__exit__()

    #override
    def onClientProcessData(self, data):
        filename_path = Path( data[0] )

        #print('printing self.count(0)...', self.count)

        done = False

        image = cv2.imread( str(filename_path) )
        if image is None:
            print ( 'Failed to extract %s, reason: cv2.imread() fail.' % ( str(filename_path) ) )
        else:
            if self.type == 'rects':
                rects = self.e.extract_from_bgr (image) # from MTCExtractor.py

                # rects is (left, top, right, bottom)

                if len(rects) > 0:
                    return [str(filename_path), rects]
                else:
                    return [str(filename_path), rects]

            elif self.type == 'landmarks':
                rects = data[1]
                landmarks = self.e.extract_from_bgr (image, rects) # from LandmarksExtractor.py
                #print(str(filename_path), landmarks)

                if self.own_video:
                    # for extracting self video - otherwise comment out
                    output = open("./workspace/data_dst.txt", "a", encoding="utf-8")
                    output.write(str([str(filename_path), landmarks]) + '\n')
                    return [str(filename_path), landmarks]
                else:
                    output = open("./workspace/data_src.txt", "a", encoding="utf-8")
                    output.write(str([str(filename_path), landmarks]) + '\n')

                    # if extracting trump video
                    '''
                    =============================================================
                    '''

                    width_arr = []
                    height_arr = []

                    for i in landmarks[0][1]:
                        width_arr.append(i[0])
                        height_arr.append(i[1])

                    max_width = max(width_arr) - min(width_arr)
                    max_height = max(height_arr) - min(height_arr)

                    #print('printing self.count...',self.count)

                    width_arr_dst = []
                    height_arr_dst = []
                    max_width_dst = 0
                    max_height_dst = 0

                    dst_landmarks = []

                    file = open("./workspace/data_dst.txt", "r", encoding="utf-8", errors = 'ignore')
                    for line in file:
                        num = int(str(line[64:69]))
                        if num == self.count:

                            width = True

                            # :(

                            a = line.split("[")[3]
                            b = a[0:len(a) - 5]
                            b = b.replace('(', '')
                            b = b.replace(',', '')
                            b = b.replace(')', '')
                            c = b.split(" ")

                            #print(c)

                            for i in c:
                                if width:
                                    width_arr_dst.append(int(i))
                                    width = False
                                else:
                                    height_arr_dst.append(int(i))
                                    width = True

                            for i in range(len(width_arr_dst)):
                                dst_landmarks.append((width_arr_dst[i], height_arr_dst[i]))

                            max_width_dst = max(width_arr_dst) - min(width_arr_dst)
                            max_height_dst = max(height_arr_dst) - min(height_arr_dst)

                    self.count += 1

                    width_ratio = float(max_width/max_width_dst)
                    height_ratio = float(max_height/max_height_dst)

                    #mouth_landmarks = landmarks[48:68]

                    print('dst_landmarks', dst_landmarks)

                    mouth_landmarks_dst = dst_landmarks[49:68]

                    landmarks2 = landmarks[0][1][0:49]
                    print(len(landmarks2))

                    for i in range(len(mouth_landmarks_dst)):
                        landmarks2.append(((int((mouth_landmarks_dst[i][0] - dst_landmarks[48][0]) * width_ratio)) +
                                           landmarks[0][1][48][0],
                                           (int((mouth_landmarks_dst[i][1] - dst_landmarks[48][1]) * height_ratio)) +
                                           landmarks[0][1][48][1]))
                    print(len(landmarks2))

                    landmarks3 = []
                    landmarks3.append((landmarks[0][0], landmarks2))

                    output2 = open("./workspace/data_src_altered.txt", "a", encoding="utf-8")
                    output2.write(str([str(filename_path), landmarks3]) + '\n')

                    '''
                    =============================================================
                    '''

                    return [str(filename_path), landmarks3]

            elif self.type == 'final':
                result = []
                faces = data[1]

                #landmarks_ori = data[1][0][1]

                # own video
                # X:\python\DF\DFL\DeepFaceLabTorrent\workspace-final1\data_dst\aligned_debug\00001_debug.png
                # last 20 are mouth?. start from landmarks[48] which is middle height leftmost for the mouth, to landmarks_ori[67]
                # maybe put bottom 5 points down the same amount landmarks_ori[66] - landmarks_ori[62] (middle mouth points) - these are landmarks_ori[6] to landmarks_ori[10]
                # can get scale of face to compare to target, total height vs total width
                #
                landmarks_ori = [(720, 532), (720, 590), (727, 641), (735, 685), (749, 735), (778, 779), (807, 815), (836, 844), (894, 866), (952, 851), (988, 822), (1024, 786), (1046, 743), (1068, 692), (1075, 641), (1082, 590), (1089, 540), (756, 489), (778, 482), (807, 475), (829, 482), (858, 489), (952, 496), (981, 489), (1003, 489), (1039, 496), (1053, 504), (901, 554), (901, 590), (894, 627), (894, 656), (865, 670), (879, 677), (894, 685), (916, 677), (930, 677), (785, 532), (807, 525), (829, 525), (850, 540), (829, 547), (807, 547), (952, 547), (974, 532), (1003, 532), (1017, 547), (1003, 554), (974, 554), (829, 743), (850, 728), (879, 721), (894, 728), (908, 728), (937, 735), (959, 750), (937, 764), (916, 779), (894, 779), (872, 772), (850, 764), (836, 743), (872, 743), (894, 743), (908, 750), (952, 750), (916, 750), (894, 750), (872, 743)]

                #red tie
                #landmarks_ori = [(577, 282), (588, 325), (594, 363), (604, 401), (615, 438), (642, 465), (675, 487), (707, 509), (756, 525), (804, 509), (831, 487), (847, 471), (869, 438), (880, 395), (885, 363), (891, 320), (891, 276), (642, 260), (664, 249), (691, 249), (712, 249), (729, 255), (804, 249), (820, 244), (842, 239), (864, 239), (874, 244), (766, 293), (766, 320), (772, 347), (772, 368), (739, 379), (750, 379), (766, 384), (783, 379), (788, 379), (669, 287), (691, 282), (707, 276), (723, 287), (707, 293), (691, 293), (799, 287), (820, 276), (837, 276), (847, 282), (837, 287), (815, 287), (712, 438), (734, 422), (756, 406), (766, 411), (777, 406), (799, 417), (804, 433), (793, 455), (777, 465), (761, 471), (745, 471), (729, 460), (718, 438), (750, 422), (766, 422), (777, 422), (804, 433), (777, 449), (761, 449), (745, 449)]

                #blue tie
                #landmarks_ori = [(474, 230), (483, 261), (492, 297), (501, 325), (515, 356), (538, 379), (560, 392), (592, 410), (632, 419), (668, 406), (686, 392), (700, 379), (718, 356), (727, 320), (732, 288), (732, 257), (732, 221), (515, 185), (538, 171), (556, 167), (574, 162), (587, 167), (655, 167), (668, 162), (686, 158), (700, 167), (714, 176), (623, 203), (623, 225), (628, 248), (628, 266), (605, 284), (614, 284), (628, 284), (637, 284), (646, 279), (542, 207), (556, 203), (569, 203), (583, 207), (574, 212), (560, 212), (650, 207), (668, 198), (682, 198), (691, 203), (682, 207), (664, 207), (592, 338), (610, 320), (623, 311), (632, 311), (641, 306), (655, 320), (664, 334), (655, 352), (646, 361), (632, 361), (619, 361), (605, 356), (596, 338), (623, 325), (632, 325), (641, 325), (664, 334), (641, 343), (632, 347), (619, 347)]

                if self.debug:
                    debug_output_file = '{}_{}'.format( str(Path(str(self.output_path) + '_debug') / filename_path.stem),  'debug.png')
                    debug_image = image.copy()

                for (face_idx, face) in enumerate(faces):
                    output_file = '{}_{}{}'.format(str(self.output_path / filename_path.stem), str(face_idx), '.png')

                    rect = face[0]
                    image_landmarks = np.array(face[1]) # source landmarks from full image

                    if self.debug:
                        facelib.LandmarksProcessor.draw_rect_landmarks (debug_image, rect, image_landmarks, self.image_size, self.face_type)

                    if self.face_type == FaceType.MARK_ONLY:
                        face_image = image
                        face_image_landmarks = image_landmarks
                    else:

                        # used for stabilization
                        #image_to_face_mat = facelib.LandmarksProcessor.get_transform_mat (landmarks_ori, self.image_size, self.face_type)

                        image_to_face_mat2 = facelib.LandmarksProcessor.get_transform_mat (image_landmarks, self.image_size, self.face_type)

                        # change below to image_to_face_mat instead of image_to_face_mat2 to stabilize
                        face_image = cv2.warpAffine(image, image_to_face_mat2, (self.image_size, self.image_size), cv2.INTER_LANCZOS4)

                        face_image_landmarks = facelib.LandmarksProcessor.transform_points (image_landmarks, image_to_face_mat2)

                    cv2.imwrite(output_file, face_image)

                    DFLPNG.embed_data(output_file, face_type = FaceType.toString(self.face_type),
                                                   landmarks = face_image_landmarks.tolist(),
                                                   yaw_value = facelib.LandmarksProcessor.calc_face_yaw (face_image_landmarks),
                                                   pitch_value = facelib.LandmarksProcessor.calc_face_pitch (face_image_landmarks),
                                                   source_filename = filename_path.name,
                                                   source_rect=  rect,
                                                   source_landmarks = image_landmarks.tolist()
                                        )

                    result.append (output_file)

                if self.debug:
                    cv2.imwrite(debug_output_file, debug_image )

                return result
        return None

        #overridable
    def onClientGetDataName (self, data):
        #return string identificator of your data
        return data[0]

    #override
    def onHostResult (self, data, result):
        if self.manual == True:
            self.landmarks = result[1][0][1]

            image = cv2.addWeighted (self.original_image,1.0,self.text_lines_img,1.0,0)
            view_rect = (np.array(self.rect) * self.view_scale).astype(np.int).tolist()
            view_landmarks  = (np.array(self.landmarks) * self.view_scale).astype(np.int).tolist()
            facelib.LandmarksProcessor.draw_rect_landmarks (image, view_rect, view_landmarks, self.image_size, self.face_type)

            if self.param['rect_locked']:
                facelib.draw_landmarks(image, view_landmarks, (255,255,0) )
            self.param['redraw_needed'] = False

            cv2.imshow (self.wnd_name, image)
            return 0
        else:
            if self.type == 'rects':
                self.result.append ( result )
            elif self.type == 'landmarks':
                self.result.append ( result )
            elif self.type == 'final':
                self.result += result

            return 1

    #override
    def onHostProcessEnd(self):
        if self.manual == True:
            cv2.destroyAllWindows()

    #override
    def get_start_return(self):
        return self.result

'''
detector
    'dlib'
    'mt'
    'manual'

face_type
    'full_face'
    'avatar'
'''
def main (input_dir, output_dir, debug, own_video, detector='mt', multi_gpu=True, cpu_only=False, manual_fix=False, manual_window_size=0, image_size=256, face_type='full_face'):
    print ("Running extractor.\r\n")

    count1 = 1

    input_path = Path(input_dir)
    output_path = Path(output_dir)
    face_type = FaceType.fromString(face_type)

    if not input_path.exists():
        print('Input directory not found. Please ensure it exists.')
        return

    if output_path.exists():
        for filename in Path_utils.get_image_paths(output_path):
            Path(filename).unlink()
    else:
        output_path.mkdir(parents=True, exist_ok=True)

    if debug:
        debug_output_path = Path(str(output_path) + '_debug')
        if debug_output_path.exists():
            for filename in Path_utils.get_image_paths(debug_output_path):
                Path(filename).unlink()
        else:
            debug_output_path.mkdir(parents=True, exist_ok=True)

    input_path_image_paths = Path_utils.get_image_unique_filestem_paths(input_path, verbose=True)
    print(input_path_image_paths)
    images_found = len(input_path_image_paths)
    faces_detected = 0
    if images_found != 0:
        if detector == 'manual':
            print ('Performing manual extract...')
            extracted_faces = ExtractSubprocessor ([ (filename,[]) for filename in input_path_image_paths ], 'landmarks', image_size, face_type, debug, own_video, cpu_only=cpu_only, manual=True, manual_window_size=manual_window_size).process()
        else:
            print ('Performing 1st pass...')
            extracted_rects = ExtractSubprocessor ([ (x,) for x in input_path_image_paths ], 'rects', image_size, face_type, debug, own_video, multi_gpu=multi_gpu, cpu_only=cpu_only, manual=False, detector=detector).process()
            #print('printing extracted_rects from 1st pass...')
            #print(extracted_rects)
            #print(type(extracted_rects))

            print ('Performing 2nd pass...')
            extracted_faces = ExtractSubprocessor (extracted_rects, 'landmarks', image_size, face_type, debug, own_video, multi_gpu=multi_gpu, cpu_only=cpu_only, manual=False).process()

            if manual_fix:
                print ('Performing manual fix...')

                if all ( np.array ( [ len(data[1]) > 0 for data in extracted_faces] ) == True ):
                    print ('All faces are detected, manual fix not needed.')
                else:
                    extracted_faces = ExtractSubprocessor (extracted_faces, 'landmarks', image_size, face_type, debug, own_video, manual=True, manual_window_size=manual_window_size).process()

        if len(extracted_faces) > 0:
            print ('Performing 3rd pass...')
            #print(extracted_faces)
            final_imgs_paths = ExtractSubprocessor (extracted_faces, 'final', image_size, face_type, debug, own_video, multi_gpu=multi_gpu, cpu_only=cpu_only, manual=False, output_path=output_path).process()
            faces_detected = len(final_imgs_paths)

    print('-------------------------')
    print('Images found:        %d' % (images_found) )
    print('Faces detected:      %d' % (faces_detected) )
    print('-------------------------')
