from moviepy.editor import VideoFileClip
from ..paths import *


def compress(name):
    clip = VideoFileClip(f"{OUTPUT_DIR}/{name}.mp4")
    clip.write_videofile(f"{OUTPUT_DIR}/compress-{name}.mp4", bitrate="5000k")
