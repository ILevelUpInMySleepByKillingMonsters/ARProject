import cv2
from ..paths import DATA_DIR
from . import ImageSegmenterProcessor

image_processor = ImageSegmenterProcessor()
cap = cv2.VideoCapture(f"{DATA_DIR}/test-2.mp4")

while cap.isOpened():
    success, frame = cap.read()

    if not success:
        break

    background, human, segment_mask = image_processor.get_segmentation(frame)
    cv2.imshow("Background", background)
    cv2.imshow("Human", human)
    cv2.imshow("Mask", segment_mask)

    key = cv2.waitKey(1)
    if key == 27:
        break
