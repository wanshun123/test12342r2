#!/usr/bin/env python

# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Google Cloud Speech API sample that demonstrates word time offsets.

Example usage:
    python transcribe_word_time_offsets.py resources/audio.raw
    python transcribe_word_time_offsets.py \
        gs://cloud-samples-tests/speech/vr.flac
"""

import argparse
import io
import glob, os, csv

def transcribe_file_with_word_time_offsets(speech_file):
    """Transcribe the given audio file synchronously and output the word time
    offsets."""
    #from google.cloud import speech
    from google.cloud import speech_v1p1beta1 as speech # for using audio_channel_count
    from google.cloud.speech import enums
    from google.cloud.speech import types
    client = speech.SpeechClient()
    
    print('speech_file parameter passed to function is ' + speech_file)
    
    files = glob.glob(speech_file)
    
    print('len of files: ' + str(len(files)))
    
    starting_number = 0
    
    filenames_and_text = []
    
    for f in files:
        print('trying with ' + str(f))
        with io.open(f, 'rb') as audio_file:
            content = audio_file.read()

        #audio = types.RecognitionAudio(content=content)
        audio = speech.types.RecognitionAudio(content=content)
        
        #config = types.RecognitionConfig(
        config_mono = speech.types.RecognitionConfig(
            encoding=enums.RecognitionConfig.AudioEncoding.LINEAR16,
            #sample_rate_hertz=16000,
            #sample_rate_hertz=48000,
            language_code='en-US',
            audio_channel_count=1, #???
            enable_word_time_offsets=True)
            
        config_stereo = speech.types.RecognitionConfig(
            encoding=enums.RecognitionConfig.AudioEncoding.LINEAR16,
            #sample_rate_hertz=16000,
            #sample_rate_hertz=48000,
            language_code='en-US',
            audio_channel_count=2, #???
            enable_word_time_offsets=True)
        
        try:
            response = client.recognize(config_mono, audio)
        except:
            response = client.recognize(config_stereo, audio)
            
        for result in response.results:
            alternative = result.alternatives[0]
            print(u'Transcript: {}'.format(alternative.transcript))
            
            # split audio file into parts of 8 words
            
            number_of_parts = int(len(alternative.words) / 8)
            
            transcriptions = []
            start_times = []
            end_times = []
                            
            for i in range(0, number_of_parts):
                end_word = alternative.words[(i * 8) + 7].word
                start_time_raw = alternative.words[i * 8].start_time
                end_time_raw = alternative.words[(i * 8) + 7].end_time
                start_time = start_time_raw.seconds + start_time_raw.nanos * 1e-9
                end_time = end_time_raw.seconds + end_time_raw.nanos * 1e-9
                
                thing = alternative.transcript.split()[i * 8:(i * 8) + 8]
                subset = thing[0] + ' ' + thing[1] + ' '  + thing[2] + ' ' + thing[3] + ' ' + thing[4] + ' ' + thing[5] + ' ' + thing[6] + ' ' + thing[7]
                transcriptions.append(subset)
                start_times.append(start_time)
                end_times.append(end_time)
                
            count = 0
                
            for i in range(len(start_times)):
                cmd = 'ffmpeg -i ' + str(f) + ' -ss ' + str(start_times[i]) + ' -t ' + str(end_times[i] - start_times[i]) + '  -acodec copy /home/ubuntu/yt/yt/data/cut/wavs/output' + str(starting_number) + '.wav'
                os.system(cmd)
                filename = 'output' + str(starting_number)
                transcription = transcriptions[i]
                filenames_and_text.append(filename + '|' + transcription)
                starting_number += 1
                
            ##files = glob.glob(os.getcwd() + '/donald-trump/*.txt')

            '''
            for i in range(len(start_times)):
                filename = 'output' + str(starting_number)
                transcription = transcriptions[i]
                filenames_and_text.append(filename + '|' + transcription)
            '''
                
        with open('metadata.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            for val in filenames_and_text:
                writer.writerow([val])


# [START speech_transcribe_async_word_time_offsets_gcs]
def transcribe_gcs_with_word_time_offsets(gcs_uri):
    """Transcribe the given audio file asynchronously and output the word time
    offsets."""
    from google.cloud import speech
    from google.cloud.speech import enums
    from google.cloud.speech import types
    client = speech.SpeechClient()

    audio = types.RecognitionAudio(uri=gcs_uri)
    config = types.RecognitionConfig(
        encoding=enums.RecognitionConfig.AudioEncoding.FLAC,
        sample_rate_hertz=16000,
        language_code='en-US',
        enable_word_time_offsets=True)

    operation = client.long_running_recognize(config, audio)

    print('Waiting for operation to complete...')
    result = operation.result(timeout=90)

    for result in result.results:
        alternative = result.alternatives[0]
        print(u'Transcript: {}'.format(alternative.transcript))
        print('Confidence: {}'.format(alternative.confidence))

        for word_info in alternative.words:
            word = word_info.word
            start_time = word_info.start_time
            end_time = word_info.end_time
            print('Word: {}, start_time: {}, end_time: {}'.format(
                word,
                start_time.seconds + start_time.nanos * 1e-9,
                end_time.seconds + end_time.nanos * 1e-9))
# [END speech_transcribe_async_word_time_offsets_gcs]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        'path', help='File or GCS path for audio file to be recognized')
    args = parser.parse_args()
    if args.path.startswith('gs://'):
        transcribe_gcs_with_word_time_offsets(args.path)
    else:
        transcribe_file_with_word_time_offsets(args.path)
