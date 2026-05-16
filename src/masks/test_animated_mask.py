import cv2
from ..paths import DATA_DIR
from .mask import ImageMaskProcessor
from ..face_detection import ImageFaceDetectorProcessor

image_processor = ImageMaskProcessor("3d_text")
face_detector = ImageFaceDetectorProcessor()

cap = cv2.VideoCapture(f"{DATA_DIR}/egon-video.mp4")

while cap.isOpened():
    success, frame = cap.read()

    if not success:
        break
    
    face_data = face_detector.detect(frame)
    mask = image_processor.mask_image(frame, face_data)

    cv2.imshow("Mask", mask)

    key = cv2.waitKey(1)
    if key == 27:
        break
