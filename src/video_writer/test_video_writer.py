import cv2
import uuid
from ..paths import DATA_DIR, OUTPUT_DIR
from ..segmentation import ImageSegmenterProcessor
from ..masks import ImageMaskProcessor
from ..utils import image_utils
from ..face_detection import ImageFaceDetectorProcessor

image_mask_processor = ImageMaskProcessor("3d_text")
image_segment_processor = ImageSegmenterProcessor()
face_detector = ImageFaceDetectorProcessor()

cap = cv2.VideoCapture(f"{DATA_DIR}/egon-video.mp4")
bg_video = cv2.VideoCapture(f"{DATA_DIR}/slav_fest.mp4")
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
first_frame = True

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
        out = cv2.VideoWriter(
            f"{OUTPUT_DIR}/{str(uuid.uuid4())}-test_video.mp4", fourcc, 24.0, (w, h)
        )
        first_frame = False

    bg_frame = image_utils.match_background_size(frame, bg_frame)

    background, human = image_segment_processor.get_segmentation(frame)
    background = image_utils.replace_no_white_background(human, bg_frame)

    face_data = face_detector.detect(human)
    mask = image_mask_processor.mask_image(background, face_data)

    out.write(mask)

out.release()
print("done")
