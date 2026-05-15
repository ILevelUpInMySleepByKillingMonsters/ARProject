import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from ..paths import SEGMENTATION_DIR


class ImageSegmenterProcessor:
    DEFAULT_MASK_COLOR = (255, 255, 255)
    DEFAULT_BG_COLOR = (0, 0, 0)

    def __init__(self, mask_color=None, bg_color=None):
        self.fg_image = None
        self.bg_image = None
        self.mask_color = mask_color or self.DEFAULT_MASK_COLOR
        self.bg_color = bg_color or self.DEFAULT_BG_COLOR
        self.segmenter = self._get_segmenter()

    def _get_segmenter(self):
        asset_path = f"{SEGMENTATION_DIR}/selfie_segmenter.tflite"
        base_options = python.BaseOptions(model_asset_path=asset_path)
        options = vision.ImageSegmenterOptions(
            base_options=base_options, output_category_mask=True, output_confidence_masks=False,
        )

        return vision.ImageSegmenter.create_from_options(options)

    def _generate_mask(self, new_image):
        segmentation_result = self.segmenter.segment(new_image)
        category_mask = segmentation_result.category_mask

        condition = category_mask.numpy_view().squeeze(-1) > 0.2

        mask = np.where(condition[..., None], self.fg_image, self.bg_image)

        _, thresh = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        return thresh

    def set_buffer(self, image):
        new_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
        image_data = new_image.numpy_view()
        self.fg_image = np.full(image_data.shape, self.mask_color, dtype=np.uint8)
        self.bg_image = np.full(image_data.shape, self.bg_color, dtype=np.uint8)

    def set_buffer_if_none(self, image):
        if self.fg_image is None or self.bg_image is None:
            self.set_buffer(image)

    def get_human(self, image):
        self.set_buffer_if_none(image)

        new_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
        thresh = self._generate_mask(new_image)

        h, w = image.shape[:2]
        thresh = self.improve_thresh(thresh, w, h)
        return cv2.bitwise_or(thresh, image)

    def get_background(self, image):
        self.set_buffer_if_none(image)

        new_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
        thresh = self._generate_mask(new_image)

        h, w = image.shape[:2]
        thresh = self.improve_thresh(thresh, w, h)
        return cv2.bitwise_and(thresh, image)

    def get_segmentation(self, image):
        h, w = image.shape[:2]

        res_image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_NEAREST)
        self.set_buffer_if_none(res_image)

        new_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=res_image)
        thresh = self._generate_mask(new_image)

        thresh = self.improve_thresh(thresh, w, h)
        return cv2.bitwise_and(thresh, image), cv2.bitwise_or(thresh, image)

    def improve_thresh(self, thresh, w, h):
        thresh = cv2.resize(thresh, (w, h), interpolation=cv2.INTER_LINEAR)

        smoothed_mask = cv2.GaussianBlur(thresh, (5, 5), 0)

        _, thresh = cv2.threshold(smoothed_mask, 127, 255, cv2.THRESH_BINARY)

        lower = (200, 200, 200)
        upper = (255, 255, 255)
        thresh = cv2.inRange(thresh, lower, upper)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        thresh = cv2.cvtColor(thresh, cv2.COLOR_BGRA2BGR)

        return thresh
