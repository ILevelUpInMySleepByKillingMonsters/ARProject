import cv2
from segment import ImageSegmenterProcessor

image_processor = ImageSegmenterProcessor()

cap = cv2.VideoCapture("../data/test-2.mp4")

while cap.isOpened():
    success, frame = cap.read()

    if not success:
        break

    human = image_processor.get_human(frame)
    cv2.imshow("Human", human)

    key = cv2.waitKey(1)
    if key == 27:
        break
