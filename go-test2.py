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

print('extracting video frames...')

import cv2
vidcap = cv2.VideoCapture('trump_cut.mp4')

# get framerate for when result images are put into a video later

video = cv2.VideoCapture("trump_cut.mp4");
fps = video.get(cv2.CAP_PROP_FPS)
video.release(); 

success,image = vidcap.read()
counter = 0
success = True

if os.path.exists("frames"):
    os.system("sudo rm -rf frames")

os.system("mkdir frames")

while success:
  cv2.imwrite("/home/paperspace/x2f/x2f/UnwrapMosaic/frames/%d.jpg" % counter, image)     # save frame as JPEG file
  success,image = vidcap.read()
  counter += 1

print('done extracting video frames')

# find and save face
import face_recognition
from PIL import Image
import glob

images = sorted(glob.glob('/home/paperspace/x2f/x2f/UnwrapMosaic/frames/*.jpg'))

if os.path.exists("faces"):
    os.system("sudo rm -rf faces")

os.system("mkdir faces")

print('seeing where face is...')

image0 = cv2.imread(images[0])
face_locations = face_recognition.face_locations(image0)
top, right, bottom, left = face_locations[0]
add_vertical = int(((bottom - top)*0.6)/2)
add_horizontal = int(((right - left)*0.6)/2)
top = int(top) - add_vertical
left = int(left) - add_horizontal
bottom = int(bottom) + add_vertical
right = int(right) + add_horizontal

print('extracting faces for each frame...')

for image in images:
    filename = image.split('/')[-1][:-4]
    image = cv2.imread(image)
    face_image = image[top:bottom, left:right]
    face_image = cv2.resize(face_image, dsize=(256, 256))
    face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
    im = Image.fromarray(face_image)
    im.save('/home/paperspace/x2f/x2f/UnwrapMosaic/faces/' + filename + '.jpg')

print('done extracting faces. Running model...')




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

# [0:40]

import subprocess
frames = int(subprocess.check_output("ffmpeg -i trump_cut.mp4 -vcodec copy -f rawvideo -y /dev/null 2>&1 | tr ^M '\n' | awk '/^frame=/ {print $2}'|tail -n 1", shell=True))

def load_img(file_path):
    img = Image.open(file_path)
    transform = Compose([Scale((256,256)), ToTensor()])
    return Variable(transform(img)).cuda()

if os.path.exists("result"):
    os.system("sudo rm -rf result")

os.system("mkdir result")

count = 0

for i in range(frames):
    driver_image = '/home/paperspace/x2f/x2f/UnwrapMosaic/faces/' + str(i) + '.jpg'
    source_imgs  = [source_path + d for d in sorted(os.listdir(source_path))][0:number_of_source_images]
    source_images = []
    for img in source_imgs:
        source_images.append(load_img(img).unsqueeze(0))
    driver_images = load_img(driver_image).unsqueeze(0)
    result = run_batch(source_images, driver_images)
    result = result.clamp(min=0, max=1)
    result = result.cpu().data
    driving_images = torchvision.utils.make_grid(driver_images.cpu().data).permute(1,2,0).numpy()
    scipy.misc.imsave('driving_images.jpg', driving_images)
    screwed_up_images = []
    for i in result:
        image = i.permute(1,2,0).numpy()
        scipy.misc.imsave('/home/paperspace/x2f/x2f/UnwrapMosaic/result/' + str(count) + '.jpg', image)
        count += 1






# turn images into video, -r sets fps

'''
os.system("sudo ldconfig")
os.system("sudo ffmpeg -r " + str(fps) + " -i /home/paperspace/x2f/x2f/UnwrapMosaic/result/%01d.jpg -vcodec mpeg4 -y result.mp4")

print('video generated (result.mp4)')

os.system("sudo ffmpeg -r " + str(fps) + " -i /home/paperspace/x2f/x2f/UnwrapMosaic/frames/%01d.jpg -vcodec mpeg4 -y result_frames.mp4")
os.system("sudo ffmpeg -r " + str(fps) + " -i /home/paperspace/x2f/x2f/UnwrapMosaic/faces/%01d.jpg -vcodec mpeg4 -y result_faces.mp4")

'''


'''
# clip video
os.system("ffmpeg -ss 1 -i video_test.mp4 -c copy -t 5 video_test_cut.mp4")
'''
