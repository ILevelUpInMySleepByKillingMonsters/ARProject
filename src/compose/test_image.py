import cv2
from ..paths import DATA_DIR
from ..segmentation import ImageSegmenterProcessor
from ..masks import ImageMaskProcessor
from ..utils import image_utils

image_mask_processor = ImageMaskProcessor()
image_segment_processor = ImageSegmenterProcessor()

img = cv2.imread(f"{DATA_DIR}/photo-egon.jpg")
bg = cv2.imread(f"{DATA_DIR}/vitebsk.jpg")

new_bg = cv2.resize(bg, (img.shape[1], img.shape[0]))

background, human = image_segment_processor.get_segmentation(img)

background = image_utils.replace_no_white_background(human, new_bg)

output = image_mask_processor.mask_image(background)
cv2.imshow("Mask", output)

key = cv2.waitKey(5000)
