import cv2
import glob
import imageio
#import matplotlib.pyplot as plt
from PIL import Image
from mlxtend.image import extract_face_landmarks
from mtcnn.mtcnn import MTCNN
from utils.DFLPNG import DFLPNG

images = glob.glob('./workspace/data_src/*.png')
detector = MTCNN()

x = 0
y = 0
w = 0
h = 0

first = True

for image in images:

    filename = image.split("/")    
    filename = str(filename[-1])
    filename = filename[9:len(filename)]
    print(filename)

    image = cv2.imread(image)
    
    source_landmarks = extract_face_landmarks(image)
    
    points_all = detector.detect_faces(image)
    
    if len(source_landmarks) > 0 and len(points_all) > 0:
        if first:
            points = points_all[0]['box'] # [x, y, width, height]
            print(points)
            
            x = points[0]
            y = points[1] - int(points[3] * 0.3) # say the hair, not covered in the face detection by default, will be 30% as high as the height of the face detected
            w = points[2]
            h = points[3] + int(points[3] * 0.6) # this will add the same height to the bottom as the top
            
            if y < 0: # face too close to top of image, can add some more to height (amount added from y)
                h = h - y
                y = 0
            
            # make width = height
            if w < h: # very likely - this will add more width to both sides of the face so the area turns square
                diff = h - w
                x = x - int(diff/2)
                w = h
            else:
                h = w
            
        first = False
                
        face = image[y:y+h,x:x+w]
        
        face = cv2.resize(face, dsize=(256, 256))
        landmarks = extract_face_landmarks(face)
            
        #print(landmarks_formatted)
        #print(landmarks.tolist())
        
        cv2.imwrite('./workspace/data_src/aligned/' + filename, face)
        
        DFLPNG.embed_data('./workspace/data_src/aligned/' + filename, face_type = 'full_face',
                                                       landmarks = landmarks.tolist(),
                                                       yaw_value = -35, # just for sorting
                                                       pitch_value = 250, # just for sorting
                                                       source_filename = filename,
                                                       source_rect = (607, 145, 894, 547), # should this matter?
                                                       source_landmarks = source_landmarks.tolist() # shouldn't matter?
                                            )
        
    '''
    imageFace = cv2.resize(imageFace, dsize=(256, 256))
    imageFace = cv2.cvtColor(imageFace, cv2.COLOR_BGR2RGB)
    im = Image.fromarray(imageFace)
    im.save('./workspace/data_src/aligned-custom/' + filename)
    '''