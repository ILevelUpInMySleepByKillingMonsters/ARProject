import cv2
from mask import ImageMaskProcessor

image_processor = ImageMaskProcessor()

cap = cv2.VideoCapture("../data/test-2.mp4")

while cap.isOpened():
    success, frame = cap.read()

    if not success:
        break

    mask = image_processor.mask_image(frame)
    cv2.imshow("Mask", mask)

    key = cv2.waitKey(1)
    if key == 27:
        break
