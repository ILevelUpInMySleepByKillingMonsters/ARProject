import cv2
import uuid
import time
from ..paths import DATA_DIR, OUTPUT_DIR
from ..segmentation import ImageSegmenterProcessor
from ..masks import ImageMaskProcessor
from ..utils import *
from ..face_detection import ImageFaceDetectorProcessor
from .video_compress import compress

image_mask_processor = ImageMaskProcessor("3d_text")
image_segment_processor = ImageSegmenterProcessor()
face_detector = ImageFaceDetectorProcessor()

cap = cv2.VideoCapture(f"{DATA_DIR}/egon-video.mp4")
bg_video = cv2.VideoCapture(f"{DATA_DIR}/slav_fest.mp4")
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
first_frame = True

progress = 0
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

out_format = ".mp4"
out_name = f"test-quality-{str(uuid.uuid4())}"
out_src = f"{OUTPUT_DIR}/{out_name}{out_format}"

total_time = time.perf_counter()

while cap.isOpened():
    success, frame = cap.read()
    success_bg, bg_frame = bg_video.read()

    if not success:
        break

    if not success_bg:
        bg_video.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    if first_frame:
        h, w = frame.shape[:2]
        out = cv2.VideoWriter(out_src + out_format, fourcc, 24.0, (w, h))
        first_frame = False

    bg_frame = image_utils.match_background_size(frame, bg_frame)

    background, human = image_segment_processor.get_segmentation(frame)
    background = image_utils.replace_no_white_background(human, bg_frame)

    face_data = face_detector.detect(human)
    mask = image_mask_processor.mask_image(background, face_data)

    out.write(mask)

    progress += 1

    print(f"{progress}/{total_frames}")

out.release()
print("done")
compress(out_name)

time_utils.show_elapsed_time(total_time)
