import cv2
import uuid
from ..paths import OUTPUT_DIR
from .video_compress import compress

class VideoWriter:
    def __init__(self, cap, size: list[int] | None = None):
        self.fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.out_format = ".mp4"
        self.out_name = f"test-quality-{str(uuid.uuid4())}"
        self.out_src = f"{OUTPUT_DIR}/{self.out_name}{self.out_format}"

        if size != None:
            self.out = cv2.VideoWriter(self.out_src, self.fourcc, 24.0, (size[0], size[1]))
        else:
            w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.out = cv2.VideoWriter(self.out_src, self.fourcc, 24.0, (w, h))

        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.progress = 0

    def write(self, frame):
        self.out.write(frame)
        self.progress += 1
        print(f"{self.progress}/{self.total_frames} - {self.progress/self.total_frames*100:.2f}%")

    def close(self):
        self.out.release()
        compress(self.out_name)