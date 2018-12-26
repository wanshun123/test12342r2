import glob
import re
import os

os.system('youtube-dl --sub-lang en --write-sub --output \'~/obamanet/data/captionstest/%(autonumber)s.%(ext)s\' --batch-file ~/obamanet/data/obama_addresses1.txt --ignore-config --extract-audio --audio-format wav --audio-quality 192K')

os.system('rename \'s/.en.vtt$/.txt/\' *.en.vtt')

transcription_files = glob.glob('/home/paperspace/obamanet/data/captionstest/*.txt')
wav_files = glob.glob('/home/paperspace/obamanet/data/captionstest/*.wav')

for file in wav_files:
    name = file.split('/')[-1]
    os.system('ffmpeg -i ' + name + ' -ar 22050 -ac 1 ' + 'n' + name) # issues with size being very low if overwriting for some reason
    os.system('rm ' + name)

count = 0
print_strings = []

for file in transcription_files:
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
        os.system('ffmpeg -i ' + wav_file + ' -ss ' + str(starts[i]) + ' -t ' + str(lengths[i]) + ' -c:v copy -c:a copy s' + str(count) + '.wav')
        print('ffmpeg -i ' + wav_file + ' -ss ' + str(starts[i]) + ' -t ' + str(lengths[i]) + ' -c:v copy -c:a copy s' + str(count) + '.wav')
        print_strings.append('ffmpeg -i ' + wav_file + ' -ss ' + str(starts[i]) + ' -t ' + str(lengths[i]) + ' -c:v copy -c:a copy s' + str(count) + '.wav')
        count += 1
    print(wav_file)
    print(text)
    print(file)
    print(transcriptions)
    print(print_strings)
            with open('metadata-test.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            for val in filenames_and_text:
                writer.writerow([val])

    '''
    wav_files = glob.glob('/home/paperspace/obamanet/data/captionstest/*.wav')
    print(wav_files)
    for i in range(len(wav_files)):
        os.system('ffmpeg -i ' + wav_files[i] + ' -ss ' + str(starts[i]) + ' -t ' + str(lengths[i]) + ' -c:v copy -c:a copy s' + str(count) + '.wav')
        count += 1
    '''

        
#ffmpeg -i obama-voice.wav -ss 00:00:05 -t 00:00:01.234 -c:v copy -c:a copy testdf.wav
ffmpeg -i obama-voice.wav -ss 00:00:05 -t 1.234 -c:v copy -c:a copy testdf.wav
