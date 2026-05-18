from .video_compress import compress
from ..paths import DATA_DIR

name = f"egon-video"

compress(name, path=DATA_DIR)
