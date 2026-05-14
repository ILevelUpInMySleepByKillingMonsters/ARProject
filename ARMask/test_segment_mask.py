import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cv2
from ARSegmentation import ImageSegmenterProcessor
from mask import ImageMaskProcessor

image_processor = ImageMaskProcessor()
image_segment_processor = ImageSegmenterProcessor()

cap = cv2.VideoCapture("../data/test-2.mp4")

while cap.isOpened():
    success, frame = cap.read()

    if not success:
        break

    # background, human = image_segment_processor.get_segmentation(frame)
    mask = image_processor.mask_image(frame)
    cv2.imshow("Mask", mask)

    key = cv2.waitKey(1)
    if key == 27:
        break
