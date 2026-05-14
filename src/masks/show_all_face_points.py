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

    image = cv2.resize(image, (w * 3, h * 3))

    h, w, _ = image.shape

    rgb_frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    detection_result = detector.detect(mp_image)

    if detection_result.face_landmarks:
        for face_landmarks in detection_result.face_landmarks:
            index = 0
            for landmark in face_landmarks:
                m_image = image
                coords = get_landmark_coords(landmark, w, h)
                point = (int(coords[0]), int(coords[1]))

                result = cv2.circle(m_image, point, 2, (255, 0, 0), -1)
                result = cv2.putText(
                    m_image,
                    str(index),
                    point,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    1,
                )

                index += 1

            cv2.imshow("Face Points", result)

    detector.close()


show_face_points()
