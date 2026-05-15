import cv2
from ..paths import DATA_DIR
from .face_detector import ImageFaceDetectorProcessor

face_detector = ImageFaceDetectorProcessor()

cap = cv2.VideoCapture(f"{DATA_DIR}/egon-video.mp4")

while cap.isOpened():
    success, frame = cap.read()

    if not success:
        break

    face_data = face_detector.detect(frame)

    cv2.imshow("Face detected", face_data.source_image)

    key = cv2.waitKey(1)
    if key == 27:
        break
