# THINGS TO CHANGE
'''
~/obamanet/data/obama_addresses.txt to a txt file of youtube url's to download
~/obamanet/data/captionstest folder to temporarily store files (also cd to this and have this script in this directory)
/home/paperspace/obamanet/data/wavs/ final data folder
'''

'''

# split whole audio
ffmpeg -i /home/paperspace/obamanet/data/captionstest/n00001.wav -filter_complex "[0:a]silencedetect=n=-35dB:d=0.4[outa]" -map [outa] -f s16le -y /dev/null |& F='-aq 70 -v warning' perl -ne 'INIT { $ss=0; $se=0; } if (/silence_start: (\S+)/) { $ss=$1; $ctr+=1; printf "ffmpeg -nostdin -i /home/paperspace/obamanet/data/captionstest/n00001.wav -ss %f -t %f $ENV{F} -y /home/paperspace/obamanet/data/obama-combined/%03d.wav\n", $se, ($ss-$se), $ctr; } if (/silence_end: (\S+)/) { $se=$1; } END { printf "ffmpeg -nostdin -i /home/paperspace/obamanet/data/captionstest/n00001.wav -ss %f $ENV{F} -y /home/paperspace/obamanet/data/obama-combined/%03d.wav\n", $se, $ctr+1; }' | bash -x

# write file names
find *.wav | sed 's:\ :\\\ :g'| sed 's/^/file /' > fl.txt

# generate silence
ffmpeg -filter_complex aevalsrc=0 -t 1 silence.wav

---

outputFile = open('fl2.txt','w')

with open('fl.txt') as fh:
    for line in fh:
        outputFile.write(line)
        outputFile.write('file silence.wav\n')

outputFile.close()

---

ffmpeg -f concat -i fl2.txt -c copy output.wav
rm fl.txt
rm fl2.txt

'''

import glob
import re
import os
import csv

os.system('youtube-dl --sub-lang en --write-sub --output \'~/obamanet/data/captionstest/%(autonumber)s.%(ext)s\' --batch-file ~/obamanet/data/obama_addresses.txt --ignore-config --extract-audio --audio-format wav --audio-quality 192K')

os.system('rename \'s/.en.vtt$/.txt/\' *.en.vtt')

transcription_files = glob.glob('/home/paperspace/obamanet/data/captionstest/*.txt')
wav_files = glob.glob('/home/paperspace/obamanet/data/captionstest/*.wav')

# make sure file is 22050 Hz
for file in wav_files:
    name = file.split('/')[-1]
    os.system('ffmpeg -i ' + name + ' -ar 22050 -ac 1 ' + 'n' + name) # issues with size being very low if overwriting for some reason
    os.system('rm ' + name)

filenames_and_text = []

for file in transcription_files:
    count = 0
    file_open = open(file, mode='r')
    text = file_open.read()
    file_open.close()
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
    wav_file = file.split('/')[-1]
    wav_file = 'n' + wav_file.split('.')[0] + '.wav'
    for i in range(len(transcriptions)):
        os.system('ffmpeg -i ' + wav_file + ' -ss ' + str(starts[i]) + ' -t ' + str(lengths[i]) + ' -c:v copy -c:a copy /home/paperspace/obamanet/data/wavs/s' + wav_file[1:6] + '-' + str(count) + '.wav')
        filenames_and_text.append('s' + wav_file[1:6] + '-' + str(count) + '|' + transcriptions[i] + '|' + transcriptions[i])
        count += 1

with open('metadata-test.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    for val in filenames_and_text:
        writer.writerow([val])

# delete original, full audio files and transcriptions (only need the metadata.csv)
wav_files = glob.glob('/home/paperspace/obamanet/data/captionstest/*.wav')
for file in wav_files:
    os.system('rm ' + file)
for file in transcription_files:
    os.system('rm ' + file)

'''
wav_files = glob.glob('/home/paperspace/obamanet/data/captionstest/*.wav')
for file in wav_files:
    os.system('ffmpeg -i ' + file + ' -af silenceremove=1:0.5:-35dB ' + file)
'''
    
#ffmpeg -i /home/paperspace/obamanet/data/captionstest/s00001-6.wav -af silenceremove=1:0.3:-45dB /home/paperspace/obamanet/data/captionstest/ss00001-6.wav
#ffmpeg-normalize /home/paperspace/obamanet/data/captionstest/s00001-6.wav -o /home/paperspace/obamanet/data/captionstest/sss00001-6.wav -c:a aac -b:a 192k

#ffmpeg -i obama-voice.wav -ss 00:00:05 -t 00:00:01.234 -c:v copy -c:a copy testdf.wav
#ffmpeg -i n00001.wav -ss 00:00:10.543 -t 2.036 -c:v copy -c:a copy testdf.wav
