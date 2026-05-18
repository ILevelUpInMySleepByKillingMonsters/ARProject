from moviepy.editor import VideoFileClip
from ..paths import *


def compress(name, format=".mp4", path=OUTPUT_DIR):
    clip = VideoFileClip(f"{path}/{name}{format}")
    clip.write_videofile(f"{path}/compress-{name}{format}", bitrate="5000k")
