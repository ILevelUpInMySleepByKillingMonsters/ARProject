import cv2
from ..paths import DATA_DIR
from ..segmentation import ImageSegmenterProcessor
from ..masks import ImageMaskProcessor
from ..utils import image_utils

image_mask_processor = ImageMaskProcessor()
image_segment_processor = ImageSegmenterProcessor()

cap = cv2.VideoCapture(f"{DATA_DIR}/test-2.mp4")
bg = cv2.imread(f"{DATA_DIR}/vitebsk.jpg")

while cap.isOpened():
    success, frame = cap.read()

    if not success:
        break

    new_bg = cv2.resize(bg, (frame.shape[1], frame.shape[0]))

    background, human = image_segment_processor.get_segmentation(frame)
    background = image_utils.replace_no_white_background(human, new_bg)

    mask = image_mask_processor.mask_image(background)
    cv2.imshow("Mask", mask)

    key = cv2.waitKey(1)
    if key == 27:
        break
