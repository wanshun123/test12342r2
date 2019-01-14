# run with python2
# https://github.com/oawiles/X2Face
# cd /home/paperspace/x2f/x2f/UnwrapMosaic

import numpy as np
import torch.nn as nn
import os
import torch
from PIL import Image
from torch.autograd import Variable
from UnwrappedFace import UnwrappedFaceWeightedAverage, UnwrappedFaceWeightedAveragePose
import torchvision
from torchvision.transforms import ToTensor, Compose, Scale
import scipy.misc
import argparse


parser = argparse.ArgumentParser(description='df')
parser.add_argument('--driving_start', type=int, default=0, help='Number of driving images')
parser.add_argument('--driving_images', type=int, default=12, help='Number of driving images')
parser.add_argument('--source_images', type=int, default=1, help='Number of driving images')
opt = parser.parse_args()

driving_start = opt.driving_start
number_of_driving_images = opt.driving_images
number_of_source_images = opt.source_images

# extract video frames 
import cv2
vidcap = cv2.VideoCapture('video_test_cut.mp4')

# get framerate for when result images are put into a video later
video = cv2.VideoCapture("video_test_cut.mp4");
fps = video.get(cv2.CAP_PROP_FPS)
video.release(); 

success,image = vidcap.read()
count = 0
success = True

if os.path.exists("frames"):
    os.system("sudo rm -rf frames")

os.system("mkdir frames")

while success:
  cv2.imwrite("/home/paperspace/x2f/x2f/UnwrapMosaic/frames/%d.jpg" % count, image)     # save frame as JPEG file
  success,image = vidcap.read()
  count += 1

print('done extracting video frames')









# find and save face
import face_recognition
from PIL import Image
import glob

images = sorted(glob.glob('/home/paperspace/x2f/x2f/UnwrapMosaic/frames/*.jpg'))

if os.path.exists("faces"):
    os.system("sudo rm -rf faces")

os.system("mkdir faces")

print('extracting faces for each frame...')

for image in images:
    filename = image.split('/')[-1][:-4]
    print(image, filename)
    image = cv2.imread(image)
    top = 74
    left = 430
    bottom = 587
    right = 943
    face_image = image[top:bottom, left:right]
    face_image = cv2.resize(face_image, dsize=(256, 256))
    face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
    im = Image.fromarray(face_image)
    im.save('/home/paperspace/x2f/x2f/UnwrapMosaic/faces/' + filename + '.jpg')

'''
    face_locations = face_recognition.face_locations(image)
    for face_location in face_locations:
        top, right, bottom, left = face_location
        add_vertical = int(((bottom - top)*0.6)/2)
        add_horizontal = int(((right - left)*0.6)/2)
        top = top - add_vertical
        left = left - add_horizontal
        bottom = bottom + add_vertical
        right = right + add_horizontal
        top = 74
        left = 430
        bottom = 587
        right = 943
        print("A face is located at pixel location Top: {}, Left: {}, Bottom: {}, Right: {}".format(top, left, bottom, right))
        face_image = image[top:bottom, left:right]
        pil_image = Image.fromarray(face_image)
        pil_image.save('/home/paperspace/x2f/x2f/UnwrapMosaic/faces/' + str(count) + '.jpg')
        count += 1
'''

print('done extracting faces. Running model...')

#print("A face is located at pixel location Top: {}, Left: {}, Bottom: {}, Right: {}".format(top, left, bottom, right))





# resize to 256x256
'''
if os.path.exists("face-resize"):
    os.system("sudo rm -rf face-resize")

os.system("mkdir face-resize")

from PIL import Image
import os, sys

path = "/home/paperspace/x2f/x2f/UnwrapMosaic/faces/"
dirs = os.listdir( path )

def resize():
    for item in dirs:
        if os.path.isfile(path+item):
            im = Image.open(path+item)
            f, e = os.path.splitext(path+item)
            imResize = im.resize((256,256), Image.ANTIALIAS)
            imResize.save("/home/paperspace/x2f/x2f/UnwrapMosaic/faces/" + item, 'JPEG', quality=90)

resize()
'''









def run_batch(source_images, pose_images, requires_grad=False, volatile=False):
    return model(pose_images, *source_images)

BASE_MODEL = '/home/paperspace/x2f/x2f/release_models/' # Change to your path
state_dict = torch.load(BASE_MODEL + 'x2face_model.pth')

model = UnwrappedFaceWeightedAverage(output_num_channels=2, input_num_channels=3, inner_nc=128)
model.load_state_dict(state_dict['state_dict'])
model = model.cuda()

model = model.eval()
#model = torch.no_grad()

driver_path = '/home/paperspace/x2f/x2f/UnwrapMosaic/faces/'
source_path = './examples/Taylor_Swift/1.6/vBgiDYBCuxY/'

driver_imgs = [driver_path + d for d in sorted(os.listdir(driver_path))][driving_start:driving_start + number_of_driving_images] # 16 driving frames
#driver_imgs = [driver_path + d for d in sorted(os.listdir(driver_path))] # all frames
source_imgs  = [source_path + d for d in sorted(os.listdir(source_path))][0:number_of_source_images] # 3 source frames

def load_img(file_path):
    img = Image.open(file_path)
    transform = Compose([Scale((256,256)), ToTensor()])
    return Variable(transform(img)).cuda()

# Driving the source image with the driving sequence
source_images = []
for img in source_imgs:
    source_images.append(load_img(img).unsqueeze(0).repeat(len(driver_imgs), 1, 1, 1))

driver_images = None
for img in driver_imgs:
    if driver_images is None:
        driver_images = load_img(img).unsqueeze(0)
    else:
        driver_images = torch.cat((driver_images, load_img(img).unsqueeze(0)), 0)

# Run the model for each
result = run_batch(source_images, driver_images)
result = result.clamp(min=0, max=1)
result = result.cpu().data

os.system("free -m")

driving_images = torchvision.utils.make_grid(driver_images.cpu().data).permute(1,2,0).numpy()
scipy.misc.imsave('driving_images.jpg', driving_images)

if os.path.exists("result"):
    os.system("sudo rm -rf result")

os.system("mkdir result")

for frame, i in enumerate(result):
    image = i.permute(1,2,0).numpy()
    scipy.misc.imsave('/home/paperspace/x2f/x2f/UnwrapMosaic/result/' + str(frame) + '.jpg', image)

print('model done. Making result video...')







# turn images into video, -r sets fps

os.system("sudo ldconfig")
os.system("sudo ffmpeg -r " + str(fps) + " -i /home/paperspace/x2f/x2f/UnwrapMosaic/result/%01d.jpg -vcodec mpeg4 -y result.mp4")

print('video generated (result.mp4)')

os.system("sudo ffmpeg -r " + str(fps) + " -i /home/paperspace/x2f/x2f/UnwrapMosaic/frames/%01d.jpg -vcodec mpeg4 -y result_frames.mp4")
os.system("sudo ffmpeg -r " + str(fps) + " -i /home/paperspace/x2f/x2f/UnwrapMosaic/faces/%01d.jpg -vcodec mpeg4 -y result_faces.mp4")




'''
# clip video
os.system("ffmpeg -ss 1 -i video_test.mp4 -c copy -t 5 video_test_cut.mp4")
'''