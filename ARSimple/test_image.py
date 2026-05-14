import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2
from ARSegmentation import ImageSegmenterProcessor
from ARMask import ImageMaskProcessor
from ARSimple.background_handler import *

image_mask_processor = ImageMaskProcessor()
image_segment_processor = ImageSegmenterProcessor()

img = cv2.imread("../data/photo-egon.jpg")

bg = cv2.imread("../data/vitebsk.jpg")

new_bg = cv2.resize(bg, (img.shape[1], img.shape[0]))

background, human = image_segment_processor.get_segmentation(img)

background = process_background(human, new_bg)

output = image_mask_processor.mask_image(background)
cv2.imshow("Mask", output)

key = cv2.waitKey(5000)
