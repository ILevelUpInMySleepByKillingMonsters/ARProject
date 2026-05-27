import cv2
from ..paths import DATA_DIR
from . import ImageQualitySegmenterProcessor

image_processor = ImageQualitySegmenterProcessor()
cap = cv2.VideoCapture(f"{DATA_DIR}/test-2.mp4")

while cap.isOpened():
    success, frame = cap.read()

    if not success:
        break

    _, human = image_processor.get_segmentation(frame)
    cv2.imshow("Human", human)

    key = cv2.waitKey(1)
    if key == 27:
        break
