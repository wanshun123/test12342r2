import os
source1 = "numpy_features"
dest11 = "numpy_features_valid"
files = os.listdir(source1)
import shutil
import numpy as np
for f in files:
    if np.random.rand(1) < 0.1:
        shutil.move(source1 + '/'+ f, dest11 + '/'+ f)


=== preprocessing
sudo python2 extract_feats2.py -w voiceloop-in-the-wild-experiments-master/data/donald-trump-silence/data/wav/donald-trump -t voiceloop-in-the-wild-experiments-master/data/donald-trump-silence/data/txt/donald-trump

=== training

sudo python2 train.py --noise 1 --expName trump_silenced_clean --seq-len 1600 --max-seq-len 1600 --data latest_features --nspk 1 --lr 1e-5 --epochs 10

sudo python2 train.py --noise 1 --expName trump_silenced_clean_training2 --seq-len 1600 --max-seq-len 1600 --data latest_features --nspk 1 --lr 1e-4 --checkpoint checkpoints/trump_silenced_clean/bestmodel.pth --epochs 90

sudo python2 generate.py  --text "I am extremely happy and excited to officially announce my candidacy to the president of the united states" --checkpoint checkpoints/trump_silenced_clean_training2/bestmodel.pth

=== deep-voice-conversion

sudo python2 train1.py proper_train_1

=== merlin

cd /home/paperspace/merlin/merlin-master/egs/build_your_own_voice/s1
cd /
cd /usr/local
cd /home/paperspace

sudo ldconfig /usr/local/cuda-9.1/lib64
sudo ldconfig /usr/local/cuda/lib64

export LD_LIBRARY_PATH=/usr/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/cuda-9.1/lib64:$LD_LIBRARY_PATH


https://github.com/CSTR-Edinburgh/merlin/archive/master.zip

./01_setup.sh my_voice
# create folder labels/label_state_align?
# go to conf/global_settings.cfg and edit path to festival
./02_prepare_labels.sh './database/wav' './database/txt' './database/labels'
sudo ./03_prepare_acoustic_features.sh './database/wav' './database/features'
sudo ./04_prepare_conf_files.sh './conf/global_settings.cfg'
sudo ./05_train_duration_model.sh './conf/duration_my_voice.conf'
06_train_acoustic_model.sh '/conf/acoustic_my_voice.conf'

=== gst-tacotron

# core dumped error
sudo python3 preprocess.py --dataset ljspeech

===

cd /home/paperspace/merlin/merlin-master/egs/build_your_own_voice/s1
cd /
cd /usr/local
cd /home/paperspace

which nvcc
nvcc -V
# nvcc file at '/usr/local/cuda/bin/'
import os; print(os.environ["PATH"])
echo $PATH

export PATH=/home/ubuntu/venv/bin:/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda:/usr/local/cuda/lib64:/opt/OpenBLAS-no-openmp/lib

export PATH=/usr/local/cuda-10.0/bin/:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-10.0/lib64:$LD_LIBRARY_PATH

.theanorc file in /home/paperspace:

[global]
device = cuda
floatX = float32
[cuda] 
root = /usr/local/cuda-9.1

exec bash

chown -R paperspace /home/paperspace/.theano
chmod -R 775 /home/paperspace/.theano

sudo ldconfig /usr/local/cuda-9.1/lib64

---

current error: you are 'tring' to use the old GPU back end, which occurs with the latest theano. If downgrading theano if suggested the old no gpu found error reoccurs. 
conda install theano=1.0

THEANO_FLAGS="mode=FAST_RUN,device=cuda0,"$MERLIN_THEANO_FLAGS
MERLIN_THEANO_FLAGS="mode=FAST_RUN,device=cuda0,cuda.root=/usr/local/cuda-9.0,floatX=float32,on_unused_input=ignore"

--- dvc2

/home/paperspace/dvc2/dvc2/model/checkpoint
data_path: '/home/paperspace/voiceloop/test/vctk/VCTK-Corpus/voiceloop-in-the-wild-experiments-master/data/donald-trump-silence/data/wav/donald-trump/*.wav'
/home/paperspace/deep-voice-conversion/deep-voice-conversion-master/data/lisa/data/timit/raw/TIMIT/TRAIN

python train2.py '/home/paperspace/dvc2/dvc2/model/checkpoint' 'trump_test_1'
python train2.py -ckpt /home/paperspace/dvc2/dvc2/model/checkpoint case2 trump_test_1

# updating nvidia drivers
sudo apt-get purge nvidia-*
sudo apt-get install nvidia-390
sudo apt-get install nvidia-410
sudo apt-get install nvidia-415 --fix-missing
# updating cuda
https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu&target_version=1604&target_type=runfilelocal
wget https://developer.nvidia.com/compute/cuda/10.0/Prod/local_installers/cuda_10.0.130_410.48_linux
sudo sh cuda_10.0.130_410.48_linux
# add to environment variable
export PATH=/usr/local/cuda-9.0/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-9.0/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
sudo .bashrc
# tensorflow nonsense
(tensorflow)$ pip2 install --upgrade pip  # for Python 2.7
(tensorflow)$ pip3 install --upgrade pip # for Python 3.n
(tensorflow)$ pip2 install --upgrade tensorflow      # for Python 2.7
(tensorflow)$ pip3 install --upgrade tensorflow     # for Python 3.n
(tensorflow)$ pip install --upgrade tensorflow-gpu  # for Python 2.7 and GPU
(tensorflow)$ pip2 install --upgrade tensorflow-gpu # for Python 3.n and GPU
(tensorflow)$ pip install --upgrade tensorflow-gpu==1.2.0 # for a specific version
sudo pip2 install tensorflow==1.5
pip2 show tensorflow

pip install --upgrade tensorflow-gpu==1.5
pip install --upgrade tensorflow==1.5
python generate_train_data.py --file trump.mp4 --num 400 --landmark-model shape_predictor_68_face_landmarks.dat

pip uninstall tensorflow
pip uninstall tensorboard
pip uninstall tensorflow-gpu

sudo apt-get install cuda-9.1

# only this avoids core dumped illegal instructions bs on tensorflow? Apparently >= 1.6 has issues with older cpu's
pip install --upgrade tensorflow-gpu==1.5.0
pip install --upgrade tensorflow==1.5.0
pip install --upgrade tensorboard==1.5.0

# ???
conda install tensorflow
$ conda create -n tensorflow
$ conda install tensorflow-gpu -n tensorflow


python train.py 1

nohup python train1.py clean_train1_4

---

# RUN THESE FIRST - otherwise libcublas.so.9.0: cannot open shared object file
export PATH=${PATH}:/usr/local/cuda-9.0/bin
export CUDA_HOME=${CUDA_HOME}:/usr/local/cuda:/usr/local/cuda-9.0
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/cuda-9.0/lib64
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/extras/CUPTI/lib64

export PATH=${PATH}:/usr/local/cuda-9.1/bin
export CUDA_HOME=${CUDA_HOME}:/usr/local/cuda:/usr/local/cuda-9.1
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/cuda-9.1/lib64
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/extras/CUPTI/lib64

export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/cuda/lib64

#EC2
export PATH=${PATH}:/usr/local/cuda-10.0/bin
export CUDA_HOME=${CUDA_HOME}:/usr/local/cuda:/usr/local/cuda-10.0
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/cuda-10.0/lib64
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/extras/CUPTI/lib64

#python 2 or python 3? should be python3. Running with or without sudo seems to run different
paperspace@psroeggij:~/home/paperspace/gsttaco/gst-tacotron-master$ python3 preprocess.py --dataset ljspeech
python3 train.py
# for https://github.com/keithito/tacotron and amazon machine
/home/ubuntu/taco/taco/pdata/data/donald-trump/data/wav
# trying to train from checkpoint
nohup python3 train.py --restore_step 1
# test model
python3 eval2.py --checkpoint '/home/paperspace/gsttaco/logs-tacotron/model.ckpt-861000'
python3 eval2.py --checkpoint '/home/paperspace/gsttaco/logs-tacotron/model.ckpt-925000'
python3 eval2.py --checkpoint '/home/paperspace/gsttaco/logs-tacotron/model.ckpt-777000'
python3 eval2.py --checkpoint '/home/paperspace/gsttaco/logs-tacotron/model.ckpt-659000'
python3 eval2.py --checkpoint '/home/paperspace/gsttaco/logs-tacotron/model.ckpt-520000'

python3 eval2.py --checkpoint '/home/paperspace/gsttaco/logs-tacotron/model.ckpt-1105000'
python3 eval2.py --checkpoint '/home/paperspace/gsttaco/logs-tacotron/model.ckpt-1077000'
python3 eval2.py --checkpoint '/home/paperspace/gsttaco/logs-tacotron/model.ckpt-1037000'

python3 eval2.py --checkpoint '/home/paperspace/gsttaco/logs-tacotron/model.ckpt-1037000'

#
python3 eval2.py --checkpoint '/home/paperspace/gsttaco/logs-tacotron-3rdpartyobamacheckpointcmudict/model.ckpt-599000'
python3 eval2.py --checkpoint '/home/paperspace/gsttaco/logs-tacotron/model.ckpt-731000'
python3 eval2.py --checkpoint '/home/paperspace/gsttaco/2/3/tacotron-master/tacotron-20180906/model.ckpt'
#
https://drive.google.com/open?id=1911tfDgjIAnXE6yKZkk8v9eGoIBqefdp
https://medium.com/tinghaochen/how-to-download-files-from-google-drive-through-terminal-4a6802707dbb
python download_gdrive.py 1EJO6QKbUXJadb4AakFBerukReNbVPeSU ~/f2f/f2f/20181229_110559.mp4

# try with both python2 and python3
paperspace@psroeggij:~/home/paperspace/dvc2/dvc2$ nohup python train1.py clean_train1_2
python eval1.py test
python train2.py github-model train2_1
# tensorflow 1.12: Aborted(core dumped), 1.7: Illegal instruction (core dumped), 
# 1.5: InvalidArgumentError (see above for traceback): No OpKernel was registered to support Op 'NcclAllReduce' with these attrs.  Registered devices: [CPU], Registered kernels: <no registered kernels>
# uninstall tensorflow and only have tensorflow-gpu: libcublas.so.9.0: cannot open shared object file: No such file or directory (need CUDA 9.0)
sudo apt-get install cuda-9-0 # doesn't work

https://askubuntu.com/questions/959835/how-to-remove-cuda-9-0-and-install-cuda-8-0-instead
https://github.com/tensorflow/tensorflow/issues/15604 #libcublas.so.9.0: cannot open shared object file: No such file or directory
https://github.com/tensorflow/tensorflow/issues/6698#issuecomment-431662397
https://medium.com/@zhanwenchen/install-cuda-and-cudnn-for-tensorflow-gpu-on-ubuntu-79306e4ac04e

export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda/lib64"

# FOR PYTHON2
CUDA 9.0 (https://developer.nvidia.com/cuda-90-download-archive?target_os=Linux)
tensorflow-gpu 1.5 (sudo pip3 install --upgrade tensorflow-gpu==1.5)
CudNN 7.0.x? (https://developer.nvidia.com/rdp/cudnn-archive)
nvidia-smi 384 (sudo apt-get install nvidia-390)



--- gst-tacotron

# gst-tacotron: code for turning custom obama, trump etc. voices into the format used by ljspeech

import glob, os, csv
files = glob.glob(os.getcwd() + '/donald-trump/*.txt')

filenames_and_text = []

for i in files:
    with open(i) as f:
        filename = os.path.basename(i)[:-4] # remove .txt at end
        text = str(f.readlines())[2:-2] # remove first 2 and last 2 characters
        filenames_and_text.append(filename + '|' + text)

with open('trump.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    for val in filenames_and_text:
        writer.writerow([val])

---

# for facebook voiceloop, trying to preprocess based on suggestions
 
import os
import glob
import librosa

cwd = os.getcwd()

files = glob.glob(cwd + '/*.wav')

for i in files:
    y, sr = librosa.load(i)
    yt, index = librosa.effects.trim(y, top_db = 15)
    print(librosa.get_duration(y), librosa.get_duration(yt))
    librosa.output.write_wav(i, yt, sr)

import time
for i in files:
    y, sr = librosa.load(i)
    yt, index = librosa.effects.trim(y, top_db = 15)
    print(librosa.get_duration(y), librosa.get_duration(yt))
    time.sleep(1)
    #librosa.output.write_wav(i, yt, sr)

---

# code from preprocessing.py in gst-tacotron

count = 0 #
futures = []
index = 1
with open('metadata-lj.csv') as f:
    for line in f:
        assert count < 10
        parts = line.strip().split('|')
        wav_path = os.path.join(os.getcwd(), 'wavs', '%s.wav' % parts[0])
        print(parts)
        count += 1
        text = parts[1]
        futures.append(executor.submit(partial(_process_utterance, out_dir, index, wav_path, text)))
        index += 1
return [future.result() for future in tqdm(futures)]

---

newlines = []

with open('metadata-real3.csv') as f:
    for line in f:
        parts = line.strip().split('|')
        #newlines.append(line[:-1] + '|' + parts[1])
        #newlines.append(parts[0] + '|' + parts[1])
        newlines.append(line)

import csv
        
with open('metadata-real2.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    for val in newlines:
        writer.writerow([val])
         

newlines = []
count = 0
with open('metadata-real3.csv') as f:
    for line in f:
        parts = line.strip().split('|')
        #newlines.append(line[:-1] + '|' + parts[1])
        #newlines.append(parts[0] + '|' + parts[1])
        newlines.append(line)
        count += line.count("|")
        if line.count("|") == 0:
            print(count)
