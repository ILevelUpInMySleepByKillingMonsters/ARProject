from ..utils import *
import cv2
from ..paths import DATA_DIR
from ..segmentation import ImageSegmenterProcessor

cap = cv2.VideoCapture(f"{DATA_DIR}/green-man.mp4")

segment_processor = ImageSegmenterProcessor()

while cap.isOpened():
    success, frame = cap.read()

    if not success:
        break

    output_segment = segment_processor.get_human(frame)
    output_without_green = image_utils.remove_green_color(frame)

    cv2.imshow("Source", frame)
    cv2.imshow("Green", output_without_green)
    cv2.imshow("Segment", output_segment)

    key = cv2.waitKey(1)
    if key == 27:
        break
