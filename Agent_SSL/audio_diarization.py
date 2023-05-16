'''
TranNhiem 2023/05
Preprocessing the Audio Data for ASR Diarization 
Pipeline: 
(Video, Podcast) Raw Audio --> Preprocessing --> Feature Extraction --> Diarization --> Speaker Embedding --> Clustering --> Speaker Identification

Reference:
https://lablab.ai/t/whisper-transcription-and-speaker-identification
'''


import os
import json
import subprocess
from googleapiclient.discovery import build

# def get_channel_videos(api_key, channel_url, max_results=10):
#     channel_id = channel_url.split('/')[-1]
#     youtube = build('youtube', 'v3', developerKey=api_key)
#     videos = []
#     next_page_token = None

#     while True:
#         request = youtube.search().list(
#             part='id',
#             channelId=channel_id,
#             maxResults=50,
#             order='date',
#             pageToken=next_page_token
#         )
#         response = request.execute()
#         videos += [item['id']['videoId'] for item in response.get('items', []) if 'videoId' in item['id']]

#         next_page_token = response.get('nextPageToken')

#         if not next_page_token or len(videos) >= max_results:
#             break

#     return videos[:max_results]


def get_channel_videos(api_key, channel_url, max_results=10):
    channel_id = channel_url.split('/')[-1]
    youtube = build('youtube', 'v3', developerKey=api_key)
    response = youtube.search().list(
        part='id',
        channelId=channel_id,
        order='date',
        maxResults=max_results
    ).execute()
    video_ids = []
    for item in response.get('items', []):
        video = item.get('id', {}).get('videoId')
        if video:
            video_ids.append(video)
    return video_ids

def download_audio_from_youtube(video_url, save_path):
    video_id = video_url.split('=')[1]
    file_path = os.path.join(save_path, f"{video_id}.mp3")
    print(f"File Path: {file_path}")

    # Check if the file already exists
    if os.path.exists(file_path):
        print(f"Skipping {video_id} - File already exists")
        return

    try:
        # Download the audio using youtube-dl
        command = f"youtube-dl --verbose -x --audio-format mp3 --audio-quality 0 -o \"{file_path}\" {video_url}"

        subprocess.call(command, shell=True)
        print(f"Downloaded {video_id} - Saved at {file_path}")
    except Exception as e:
        print(f"Error downloading {video_id}: {str(e)}")

# Example usage:
api_key = "AIzaSyCE3ooWjZX4CxhfJAmKrzc0k3sCiG9AQYE"
channel_url = "https://www.youtube.com/@bigthink"
video_url = None
save_path = "/media/rick/f7a9be3d-25cd-45d6-b503-7cb8bd32dbd5/Audio/"
if not os.path.exists(save_path):
    os.makedirs(save_path)

max_results = 4

video_ids = get_channel_videos(api_key, channel_url, max_results)

for video_id in video_ids:
    video_url = f"https://www.youtube.com/watch?v={video_id}"
    download_audio_from_youtube(video_url, save_path)

if video_url:
    download_audio_from_youtube(video_url, save_path)

##-----------------Download Audio from Spotify ------------------------------
# import spotipy
# from spotipy.oauth2 import SpotifyClientCredentials
# import os
# import urllib.request

# # Spotify API credentials
# client_id = 'YOUR_CLIENT_ID'
# client_secret = 'YOUR_CLIENT_SECRET'

# # Playlist URI (e.g., 'spotify:playlist:PLAYLIST_ID')
# playlist_uri = 'SPOTIFY_PLAYLIST_URI'

# # Output directory to save audio files
# output_directory = '/path/to/output/directory'

# # Authenticate with Spotify API
# client_credentials_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
# sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

# # Get playlist tracks
# results = sp.playlist_items(playlist_uri)
# tracks = results['items']

# # Download audio files
# for track in tracks:
#     track_name = track['track']['name']
#     track_artist = track['track']['artists'][0]['name']
#     track_preview_url = track['track']['preview_url']

#     if track_preview_url:
#         # Create output filename
#         filename = f'{track_artist} - {track_name}.mp3'
#         file_path = os.path.join(output_directory, filename)

#         # Download audio file
#         urllib.request.urlretrieve(track_preview_url, file_path)
#         print(f'Downloaded: {filename}')
#     else:
#         print(f'Skipped: {track_artist} - {track_name} (No preview available)')
