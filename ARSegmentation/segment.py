import os
import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class ImageSegmenterProcessor:
    DEFAULT_MASK_COLOR = (255, 255, 255)
    DEFAULT_BG_COLOR = (0, 0, 0)

    def __init__(self, model_type="selfie_segmenter", mask_color=None, bg_color=None):
        self.mask_color = mask_color or self.DEFAULT_MASK_COLOR
        self.bg_color = bg_color or self.DEFAULT_BG_COLOR
        self.segmenter = self._get_segmenter(model_type)

    def _get_segmenter(self, model_type):
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))

        model_paths = {
            "deeplab": os.path.join(BASE_DIR, "deeplab_v3.tflite"),
            "selfie_segmenter": os.path.join(BASE_DIR, "selfie_segmenter.tflite")
        }

        asset_path = model_paths.get(model_type, model_paths["selfie_segmenter"])
        base_options = python.BaseOptions(model_asset_path=asset_path)
        options = vision.ImageSegmenterOptions(
            base_options=base_options, output_category_mask=True
        )
        
        return vision.ImageSegmenter.create_from_options(options)

    def _generate_mask(self, new_image):
        segmentation_result = self.segmenter.segment(new_image)
        category_mask = segmentation_result.category_mask

        image_data = new_image.numpy_view()

        fg_image = np.zeros(image_data.shape, dtype=np.uint8)
        fg_image[:] = self.mask_color

        bg_image = np.zeros(image_data.shape, dtype=np.uint8)
        bg_image[:] = self.bg_color

        condition = category_mask.numpy_view().squeeze(-1) > 0.2
        mask = np.where(condition[..., None], fg_image, bg_image)

        _, thresh = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)
        return thresh

    def get_human(self, image):
        new_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
        thresh = self._generate_mask(new_image)
        return cv2.bitwise_or(thresh, image)

    def get_background(self, image):
        new_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
        thresh = self._generate_mask(new_image)
        return cv2.bitwise_and(thresh, image)

    def get_segmentation(self, image):
        new_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
        thresh = self._generate_mask(new_image)
        return cv2.bitwise_and(thresh, image), cv2.bitwise_or(thresh, image)
