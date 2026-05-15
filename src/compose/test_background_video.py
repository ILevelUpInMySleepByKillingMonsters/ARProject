import cv2
from ..paths import DATA_DIR
from ..segmentation import ImageSegmenterProcessor
from ..masks import ImageMaskProcessor
from ..utils import image_utils

image_mask_processor = ImageMaskProcessor()
image_segment_processor = ImageSegmenterProcessor()

cap = cv2.VideoCapture(f"{DATA_DIR}/test-2.mp4")
bg_video = cv2.VideoCapture(f"{DATA_DIR}/green-man.mp4")


while cap.isOpened():
    success, frame = cap.read()
    _, bg_frame = bg_video.read()

    if not success:
        break

    bg_frame = image_utils.match_background_size(frame, bg_frame)

    background, human = image_segment_processor.get_segmentation(frame)
    background = image_utils.replace_no_white_background(human, bg_frame)

    cv2.imshow("Mask", background)

    key = cv2.waitKey(1)
    if key == 27:
        break
