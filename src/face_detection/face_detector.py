import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from ..paths import FACE_DETECTION
from .face_detector_config import *


class ImageFaceDetectorProcessor:
    DEFAULT_MASK_COLOR = (255, 255, 255)
    DEFAULT_BG_COLOR = (0, 0, 0)

    def __init__(self, mask_color=None, bg_color=None):
        self.fg_image = None
        self.bg_image = None
        self.mask_color = mask_color or self.DEFAULT_MASK_COLOR
        self.bg_color = bg_color or self.DEFAULT_BG_COLOR
        self.detector = self._get_segmenter()

    def _get_segmenter(self):
        asset_path = f"{FACE_DETECTION}/blaze_face_full_range_sparse.tflite"
        base_options = python.BaseOptions(model_asset_path=asset_path)
        options = vision.FaceDetectorOptions(base_options=base_options)

        return vision.FaceDetector.create_from_options(options)

    def detect(self, image):
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
        face_detector_result = self.detector.detect(mp_image)

        face_data = FaceDetectorData(source_image=image)

        if face_detector_result:
            for detection in face_detector_result.detections:
                bounding_box: BoundingBox = detection.bounding_box
                origin_x = bounding_box.origin_x
                origin_y = bounding_box.origin_y
                width = bounding_box.width
                height = bounding_box.height

                x2 = origin_x + width
                y2 = origin_y + height

                # color = (0, 255, 0)
                # thickness = 2
                # rect = cv2.rectangle(image, (origin_x, origin_y), (x2, y2), color, thickness)

                rect = image[origin_y:y2, origin_x:x2]
                face_data.source_image = rect
                face_data.bounding_box = bounding_box

        return face_data
