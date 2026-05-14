import cv2
import numpy as np
import mediapipe as mp
from ..paths import *
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

IMAGE_PATH = f"{DATA_DIR}/face.jpg"

base_options = python.BaseOptions(model_asset_path=f"{MASKS_DIR}/face_landmarker.task")
options = vision.FaceLandmarkerOptions(
    base_options=base_options, running_mode=vision.RunningMode.IMAGE, num_faces=1
)
detector = vision.FaceLandmarker.create_from_options(options)


def get_landmark_coords(face_landmarks, w, h):
    return np.array([face_landmarks.x * w, face_landmarks.y * h])


def show_face_points():
    image = cv2.imread(IMAGE_PATH)

    h, w, _ = image.shape

    rgb_frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    detection_result = detector.detect(mp_image)

    if detection_result.face_landmarks:
        for face_landmarks in detection_result.face_landmarks:
            index = 0
            while True:
                print(index)
                m_image = image.copy()
                coords = get_landmark_coords(face_landmarks[index], w, h)
                point = (int(coords[0]), int(coords[1]))

                result = cv2.circle(m_image, point, 2, (255, 0, 0), -1)
                result = cv2.putText(
                    m_image,
                    str(index),
                    point,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.3,
                    (255, 255, 255),
                    1,
                )

                cv2.imshow("Face Points", result)

                key = cv2.waitKey(0) & 0xFF

                if key == ord("q"):
                    detector.close()
                    break
                elif key == ord("1"):
                    index += 1
                elif key == ord("2"):
                    index -= 1
                else:
                    index += 1

                if index >= len(face_landmarks):
                    index = 0

                if index < 0:
                    index = len(face_landmarks) - 1

    detector.close()


show_face_points()
