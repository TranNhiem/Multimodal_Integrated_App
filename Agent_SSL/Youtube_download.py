import re
from pytube import YouTube
import subprocess
import os

def download_video(url, filelocation):
    try:
        yt = YouTube(url)
        stream = yt.streams.first()
        string = ''.join([i for i in re.findall('[\w +/.]', yt.title) if i.isalpha()])
        filename = filelocation+'/'+string+'.mp4'
        stream.download(output_path=filelocation, filename=string)
        print('Video Download completed!')
    except Exception as e:
        print(e)


# def download_audio(url, filelocation, audio_format='mp3'):
#     try:
#         yt = YouTube(url)
#         audio = yt.streams.filter(only_audio=True).first()
#         string = ''.join([i for i in re.findall('[\w +/.]', yt.title) if i.isalpha()])
#         filename = filelocation+'/'+string+'.'+audio_format
#         audio.download(output_path=filelocation, filename=string)
#         print('Audio Download completed!')
        
#         if audio_format != 'wav':
#             input_file = os.path.join(filelocation, f'{string}.{audio_format}')
#             output_file = os.path.join(filelocation, f'{string}.wav')
#             convert_to_wav(input_file, output_file)
#     except Exception as e:
#         print(e)

def download_audio(url, filelocation, audio_format='mp3'):
    try:
        yt = YouTube(url)
        audio = yt.streams.filter(only_audio=True).first()
        string = ''.join([i for i in re.findall('[\w +/.]', yt.title) if i.isalpha()])
        filename = f"{filelocation}/{string}.{audio_format}"
        audio.download(output_path=filelocation, filename=string)

        # Check if the downloaded file has the correct format
        # and rename it if necessary
        if audio.subtype != audio_format:
            original_file = f"{filelocation}/{string}"#.{audio.subtype}
            new_file = f"{filelocation}/{string}.{audio_format}"
            os.rename(original_file, new_file)

        print('Audio Download completed!')
    except Exception as e:
        print(e)



def convert_to_wav(input_file, output_file):
    try:
        subprocess.run(['ffmpeg', '-i', input_file, '-acodec', 'pcm_s16le', '-ar', '44100', '-ac', '2', output_file], check=True)
        print('Audio converted to WAV format successfully.')
    except subprocess.CalledProcessError as e:
        print(f'Error converting audio to WAV format: {e}')


def start_video_download():
    filelocation = input("Enter the file location to save the video: ")
    url = input("Enter the YouTube video URL: ")
    video_id = re.search(r'(?:v=|\/)([0-9A-Za-z_-]{11})', url)
    if video_id:
        video_id = video_id.group(1)
        download_video(f"https://www.youtube.com/watch?v={video_id}", filelocation)
    else:
        print("Invalid YouTube URL")


def start_audio_download():
    filelocation = input("Enter the file location to save the audio: ")
    url = input("Enter the YouTube video URL: ")
    audio_format = input("Enter the desired audio format (default: mp3): ") or 'mp3'
    video_id = re.search(r'(?:v=|\/)([0-9A-Za-z_-]{11})', url)
    if video_id:
        video_id = video_id.group(1)
        download_audio(f"https://www.youtube.com/watch?v={video_id}", filelocation, audio_format)
    else:
        print("Invalid YouTube URL")


# # Define Video to download
# video_url = "https://www.youtube.com/watch?v=6OozhhI6U4g&t=1527s"  #'https://www.youtube.com/watch?v=sitHS6UDMJc'
# save_path = './'#"/media/rick/f7a9be3d-25cd-45d6-b503-7cb8bd32dbd5/Audio"
# format = 'wav'  # mp3, wav, ogg, m4a, wma, flv, webm, 3gp

# # Download audio from video URL
# start_audio_download()
