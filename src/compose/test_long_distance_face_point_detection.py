import cv2
from ..paths import DATA_DIR
from ..face_detection import ImageFaceDetectorProcessor
from ..masks import ImageMaskProcessor

face_detector = ImageFaceDetectorProcessor()
mask = ImageMaskProcessor()

cap = cv2.VideoCapture(f"{DATA_DIR}/egon-video.mp4")

while cap.isOpened():
    success, frame = cap.read()

    if not success:
        break

    face_data = face_detector.detect(frame)

    face_points = mask.mask_image(frame, face_data)

    cv2.imshow("Face points", face_points)

    key = cv2.waitKey(1)
    if key == 27:
        break
