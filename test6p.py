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
            #sample_rate_hertz=4number_of_words000,
            language_code='en-US',
            audio_channel_count=1, #???
            enable_word_time_offsets=True)
            
        config_stereo = speech.types.RecognitionConfig(
            encoding=enums.RecognitionConfig.AudioEncoding.LINEAR16,
            #sample_rate_hertz=16000,
            #sample_rate_hertz=4number_of_words000,
            language_code='en-US',
            audio_channel_count=2, #???
            enable_word_time_offsets=True)
        
        try:
            response = client.recognize(config_mono, audio)
        except:
            response = client.recognize(config_stereo, audio)
            
        number_of_words = int(args.number_of_words)
            
        for result in response.results:
            print('new result in response.results...', end = '\n\n')
            alternative = result.alternatives[0]
            print(u'Transcript: {}'.format(alternative.transcript))
                        
            number_of_parts = int(len(alternative.words) / number_of_words)
            
            remaining_words = len(alternative.words) - (number_of_words * number_of_parts)
            
            print(str(number_of_parts), str(remaining_words))
                        
            transcriptions = []
            start_times = []
            end_times = []
                                        
            for i in range(0, number_of_parts):
                end_word = alternative.words[(i * number_of_words) + number_of_words - 1].word
                start_time_raw = alternative.words[i * number_of_words].start_time
                end_time_raw = alternative.words[(i * number_of_words) + number_of_words - 1].end_time
                start_time = start_time_raw.seconds + start_time_raw.nanos * 1e-9
                end_time = end_time_raw.seconds + end_time_raw.nanos * 1e-9
                
                thing = alternative.transcript.split()[i * number_of_words:(i * number_of_words) + number_of_words]
                subset = ''
                for i in range(len(thing)):
                    subset = subset + thing[i] + ' '
                subset = subset[:-1]
                transcriptions.append(subset)
                start_times.append(start_time)
                end_times.append(end_time)
                
            #count = 0
                
            for i in range(len(start_times)):
                if end_times[i] - start_times[i] > 2 and len(transcriptions[i]) > 20: # don't want blank or single word recordings
                    cmd = 'ffmpeg -i ' + str(f) + ' -ss ' + str(start_times[i]) + ' -t ' + str(end_times[i] - start_times[i]) + '  -acodec copy /home/paperspace/gsttaco/gst-tacotron-master/obama/data/wavs/output' + str(starting_number) + '.wav'
                    os.system(cmd)
                    filename = 'output' + str(starting_number)
                    transcription = transcriptions[i]
                    filenames_and_text.append(filename + '|' + transcription)
                    starting_number += 1
                #count += 1
                
            # do remainder
            
            count = number_of_parts * number_of_words
            
            if remaining_words > 0:
                end_word = alternative.words[len(alternative.words) - 1].word
                start_time_raw = alternative.words[count].start_time
                end_time_raw = alternative.words[len(alternative.words) - 1].end_time
                start_time = start_time_raw.seconds + start_time_raw.nanos * 1e-9
                end_time = end_time_raw.seconds + end_time_raw.nanos * 1e-9
                
                thing = alternative.transcript.split()[count:len(alternative.words)]
                print('thing for remaining words:', end = '\n\n')
                print(thing)
                subset = ''
                for i in range(len(thing)):
                    subset = subset + thing[i] + ' '
                subset = subset[:-1]
                transcriptions.append(subset)
                start_times.append(start_time)
                end_times.append(end_time)
                
                if end_time - start_time > 2 and len(transcriptions[number_of_parts]) > 20:
                    cmd = 'ffmpeg -i ' + str(f) + ' -ss ' + str(start_time) + ' -t ' + str(end_time - start_time) + '  -acodec copy /home/paperspace/gsttaco/gst-tacotron-master/obama/data/wavs/output' + str(starting_number) + '.wav'
                    os.system(cmd)
                    filename = 'output' + str(starting_number)
                    transcription = transcriptions[number_of_parts]
                    filenames_and_text.append(filename + '|' + transcription)
                    starting_number += 1
                
            ##files = glob.glob(os.getcwd() + '/donald-trump/*.txt')

        with open('metadata.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            for val in filenames_and_text:
                writer.writerow([val])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('path') # directory to run transcriptions on and cut up
    parser.add_argument('number_of_words', default = 12) # words per audio file
    args = parser.parse_args()
    transcribe_file_with_word_time_offsets(args.path)
