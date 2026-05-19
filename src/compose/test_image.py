import cv2
from ..paths import DATA_DIR
from ..segmentation import ImageSegmenterProcessor
from ..masks import ImageMaskProcessor
from ..utils import image_utils

image_mask_processor = ImageMaskProcessor()
image_segment_processor = ImageSegmenterProcessor()

img = cv2.imread(f"{DATA_DIR}/photo-egon.jpg")
bg = cv2.imread(f"{DATA_DIR}/vitebsk.jpg")

new_bg = image_utils.match_background_size(img, bg)

background, human, segment_mask = image_segment_processor.get_segmentation(img)

background = image_utils.overlay_background(human, new_bg, segment_mask)

output = image_mask_processor.mask_image(background)
cv2.imshow("Mask", output)

key = cv2.waitKey(5000)
