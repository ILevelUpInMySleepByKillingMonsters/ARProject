import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2
from ARSegmentation import ImageSegmenterProcessor
from ARMask import ImageMaskProcessor
from ARSimple.background_handler import *

image_mask_processor = ImageMaskProcessor()
image_segment_processor = ImageSegmenterProcessor()

cap = cv2.VideoCapture("../data/test-2.mp4")

bg = cv2.imread("../data/vitebsk.jpg")

while cap.isOpened():
    success, frame = cap.read()

    if not success:
        break

    new_bg = cv2.resize(bg, (frame.shape[1], frame.shape[0]))

    background, human = image_segment_processor.get_segmentation(frame)

    background = process_background(human, new_bg)

    mask = image_mask_processor.mask_image(background)
    cv2.imshow("Mask", mask)

    key = cv2.waitKey(1)
    if key == 27:
        break
