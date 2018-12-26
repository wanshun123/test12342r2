import glob
import re
import os

#youtube-dl --sub-lang en --skip-download --write-sub --output '~/obamanet/data/captionstest/%(autonumber)s.%(ext)s' --batch-file ~/obamanet/data/obama_addresses1.txt --ignore-config
youtube-dl --sub-lang en --write-sub --output '~/obamanet/data/captionstest/%(autonumber)s.%(ext)s' --batch-file ~/obamanet/data/obama_addresses1.txt --ignore-config --extract-audio --audio-format wav --audio-quality 192K

rename 's/.en.vtt$/.txt/' *.en.vtt

#python2 YouTube_to_WAV3.py https://www.youtube.com/watch?v=u2ZynkD3N_k

transcription_files = glob.glob('/home/paperspace/obamanet/data/captionstest/*.txt')
wav_files = glob.glob('/home/paperspace/obamanet/data/captionstest/*.wav')

for file in wav_files:
    name = file.split('/')[-1]
    os.system('ffmpeg -i ' + name + ' -ar 22050 ' + 'n' + name)
    ffmpeg -i /home/paperspace/obamanet/data/captionstest/00002.wav -ar 22050 /home/paperspace/obamanet/data/captionstest/new.wav

rm 00002.wav

#os.system("some_command with args")

for file in files:
    os.system('ffmpeg -i ' + file + ' -ar 22050 ' + 'n' + file)

ffmpeg -i 1.wav -ar 22050 new.wav
rm 1.wav

#f = open('test.txt', 'r')
#x = f.readlines()
#f.close()

file = open('test.txt',mode='r')
text = file.read()
file.close()

text = text.replace('WEBVTT\nKind: captions\nLanguage: en', '')
text = text.replace('The President: ', '')
text = text.replace('Hello, everybody.', 'Hi, everybody.')
text = text.replace('\n\n', '##') # this will represent the end of a transcription
text = text.replace('\n', ' ')

time_regex = re.compile("[0-9][0-9]:[0-9][0-9]:[0-9][0-9].[0-9][0-9][0-9]")
times = time_regex.findall(text)

start = True
starts = []
lengths = []

for time in times:
    if start:
        starts.append(time)
        start = False
    else:
        end_hours = int(time[0:2])
        end_minutes = int(time[3:5])
        end_seconds = float(time[6:12])
        end_time_total = (end_hours * 3600) + (end_minutes * 60) + end_seconds
        
        start_hours = int(starts[-1][0:2])
        start_minutes = int(starts[-1][3:5])
        start_seconds = float(starts[-1][6:12])
        start_total = (start_hours * 3600) + (start_minutes * 60) + start_seconds
        
        duration = round(end_time_total - start_total, 3)
        
        lengths.append(duration)
        
        start = True


transcription_regex = re.compile("(?<=[0-9][0-9]:[0-9][0-9]:[0-9][0-9].[0-9][0-9][0-9] --> [0-9][0-9]:[0-9][0-9]:[0-9][0-9].[0-9][0-9][0-9] )([^##]+)(?=##)")
transcriptions = transcription_regex.findall(text)
        
#ffmpeg -i obama-voice.wav -ss 00:00:05 -t 00:00:01.234 -c:v copy -c:a copy testdf.wav
