import cv2
from ..paths import DATA_DIR
from .mask import ImageMaskProcessor

image_processor = ImageMaskProcessor(show_face_points=True)

image = cv2.imread(f"{DATA_DIR}/photo-egon.jpg")

mask = image_processor.mask_image(image)
cv2.imshow("Mask", mask)

key = cv2.waitKey(5000)
