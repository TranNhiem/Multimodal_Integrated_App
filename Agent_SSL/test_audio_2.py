import os
import youtube_dl

def download_audio_from_channel(channel_urls, save_path):
    for channel_url in channel_urls:
        video_urls = get_video_urls(channel_url)
        for video_url in video_urls:
            download_audio(video_url, save_path)

def get_video_urls(channel_url):
    ydl_opts = {
        'ignoreerrors': True,
        'extract_flat': True,
        'playlist_items': '1-1000',  # Adjust the range to fetch more videos if needed
    }
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(channel_url, download=False)
        video_urls = [entry['url'] for entry in info['entries'] if entry]
    return video_urls



def download_audio(video_url, save_path):
    ydl_opts = {
        'outtmpl': os.path.join(save_path, '%(id)s.%(ext)s'),
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
    }
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        ydl.download([video_url])
        print(f"Downloaded audio from video: {video_url}")

if __name__ == "__main__":
    channel_urls = [
        "https://www.youtube.com/channel/UCSHZKyawb77ixDdsGog4iWA",
        # "https://www.youtube.com/channel/CHANNEL_ID2",
        # "https://www.youtube.com/channel/CHANNEL_ID3",
        # Add more channel URLs as needed
    ]
    save_path = "/media/rick/f7a9be3d-25cd-45d6-b503-7cb8bd32dbd5/Audio"

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    download_audio_from_channel(channel_urls, save_path)
