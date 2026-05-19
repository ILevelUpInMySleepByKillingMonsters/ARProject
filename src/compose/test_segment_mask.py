import cv2
from ..paths import DATA_DIR
from ..segmentation import ImageSegmenterProcessor
from ..masks import ImageMaskProcessor

image_mask_processor = ImageMaskProcessor()
image_segment_processor = ImageSegmenterProcessor()

cap = cv2.VideoCapture(f"{DATA_DIR}/test-2.mp4")

while cap.isOpened():
    success, frame = cap.read()

    if not success:
        break

    background, human, _ = image_segment_processor.get_segmentation(frame)
    mask = image_mask_processor.mask_image(human)

    cv2.imshow("Mask", mask)

    key = cv2.waitKey(1)
    if key == 27:
        break
