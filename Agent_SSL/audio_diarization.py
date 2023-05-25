'''
TranNhiem 2023/05
Preprocessing the Audio Data for ASR Diarization 
Pipeline: 
(Video, Podcast) Raw Audio --> Preprocessing --> Feature Extraction --> Diarization --> Speaker Embedding --> Clustering --> Speaker Identification

Reference:
https://lablab.ai/t/whisper-transcription-and-speaker-identification
'''

from Youtube_download import  download_audio

## Define Video want to download
video_url="https://www.youtube.com/watch?v=6OozhhI6U4g&t=1527s" #'https://www.youtube.com/watch?v=sitHS6UDMJc'
save_path="/media/rick/f7a9be3d-25cd-45d6-b503-7cb8bd32dbd5/Audio"
format='mp3' # mp3, wav, ogg, m4a, wma, flv, webm, 3gp
## download audio from video url

download_audio(video_url, save_path, format)

##-------------------Audio Preprocessing--------------------------
from pydub.utils import mediainfo

# file_info = mediainfo("/media/rick/f7a9be3d-25cd-45d6-b503-7cb8bd32dbd5/Audio/OpenAssistantInferenceBackendDevelopmentHandsOnCoding.mp3")
# print(file_info["format_name"])

# from pydub import AudioSegment

# audio = AudioSegment.from_file("/media/rick/f7a9be3d-25cd-45d6-b503-7cb8bd32dbd5/Audio/OpenAssistantInferenceBackendDevelopmentHandsOnCoding.mp3", format="mp4")
# audio.export("output.wav", format="wav")


##-------------------Segment the Audio--------------------------
from pydub import AudioSegment
## Get Audio Path 
audio_path=None # Update path_join(save_path, title + "." + format)
spacermilli = 2000
spacer = AudioSegment.silent(duration=spacermilli)
# audio = AudioSegment.from_wav("./output.wav") #lecun1.wav
# audio = spacer.append(audio, crossfade=0)
# audio.export('audio.wav', format='wav')

##-------------------Pyannote's Diarization --------------------------
'''
pyannote.audio also comes with pretrained models and pipelines covering a wide range of domains for voice activity detection, 
speaker segmentation, overlapped speech detection

'''
from pyannote.audio import Model

from pyannote.audio import Pipeline
weight_path="/media/rick/f7a9be3d-25cd-45d6-b503-7cb8bd32dbd5/pretrained_weights/pyannote/"
if not os.path.exists(weight_path): 
    os.makedirs(weight_path)
pipeline = Pipeline.from_pretrained('pyannote/speaker-diarization', use_auth_token=True)

DEMO_FILE = {'uri': 'blabla', 'audio': 'audio.wav'}
dz = pipeline(DEMO_FILE)  

with open("diarization.txt", "w") as text_file:
    text_file.write(str(dz))

print(*list(dz.itertracks(yield_label = True))[:10], sep="\n")

##-------------------Speaker Embedding--------------------------
def millisec(timeStr):
  spl = timeStr.split(":")
  s = (int)((int(spl[0]) * 60 * 60 + int(spl[1]) * 60 + float(spl[2]) )* 1000)
  return s

import re
dzs = open('diarization.txt').read().splitlines()

groups = []
g = []
lastend = 0
for d in dzs:   
  if g and (g[0].split()[-1] != d.split()[-1]):      #same speaker
    groups.append(g)
    g = []
  
  g.append(d)
  
  end = re.findall('[0-9]+:[0-9]+:[0-9]+\.[0-9]+', string=d)[1]
  end = millisec(end)
  if (lastend > end):       #segment engulfed by a previous segment
    groups.append(g)
    g = [] 
  else:
    lastend = end
if g:
  groups.append(g)

print(*groups, sep='\n')

# audio = AudioSegment.from_wav("audio.wav")
# gidx = -1

# for g in groups:
#   start = re.findall('[0-9]+:[0-9]+:[0-9]+\.[0-9]+', string=g[0])[0]
#   end = re.findall('[0-9]+:[0-9]+:[0-9]+\.[0-9]+', string=g[-1])[1]
#   start = millisec(start) #- spacermilli
#   end = millisec(end)  #- spacermilli
#   print(start, end)
#   gidx += 1
#   audio[start:end].export(str(gidx) + '.wav', format='wav')