import cv2
from segment import ImageSegmenterProcessor

image_processor = ImageSegmenterProcessor()

cap = cv2.VideoCapture("../data/test-2.mp4")

while cap.isOpened():
    success, frame = cap.read()

    if not success:
        break

    background = image_processor.get_background(frame)
    cv2.imshow("Background", background)

    key = cv2.waitKey(1)
    if key == 27:
        break
